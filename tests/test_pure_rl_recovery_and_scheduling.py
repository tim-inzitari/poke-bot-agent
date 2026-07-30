from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot.pure_rl.eval_public import OFFICIAL_BASELINE_IDS, aggregate_heldout_wr
from scripts import train_pure_rl


def test_exact_regression_rollback_uses_runtime_activated_anchor(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "iter26.pt"
    activated_path = tmp_path / "iter26_matchup.pt"
    behavior_path = tmp_path / "iter29.pt"
    parent_path.write_bytes(b"protected exact anchor")
    activated_path.write_bytes(b"protected exact anchor plus runtime adapters")
    behavior_path.write_bytes(b"current behavior")
    parent = train_pure_rl._verified_checkpoint_identity(
        {
            "path": str(parent_path),
            "digest": "sha256:"
            + hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        }
    )
    activated = train_pure_rl._verified_checkpoint_identity(
        {
            "path": str(activated_path),
            "digest": "sha256:"
            + hashlib.sha256(activated_path.read_bytes()).hexdigest(),
        }
    )
    behavior = train_pure_rl._verified_checkpoint_identity(
        {
            "path": str(behavior_path),
            "digest": "sha256:"
            + hashlib.sha256(behavior_path.read_bytes()).hexdigest(),
        }
    )
    receipt = tmp_path / "runtime-boundary.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_runtime_boundary_activation/v1",
                "parent_learner": parent.as_dict(),
                "activated_learner": activated.as_dict(),
            }
        ),
        encoding="utf-8",
    )
    chosen, source = train_pure_rl._exact_regression_rollback_identity(
        {
            "matchup_runtime_activation": {
                "receipt": str(receipt),
                "learner_digest": activated.digest,
            }
        },
        exact_anchor=parent,
        behavior_before=behavior,
    )
    assert chosen == activated
    assert source == "runtime_activated_exact_anchor"


def test_exact_regression_rollback_preserves_behavior_if_receipt_is_invalid(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.pt"
    behavior_path = tmp_path / "behavior.pt"
    anchor_path.write_bytes(b"anchor")
    behavior_path.write_bytes(b"behavior")
    anchor = train_pure_rl._verified_checkpoint_identity(
        {
            "path": str(anchor_path),
            "digest": "sha256:"
            + hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        }
    )
    behavior = train_pure_rl._verified_checkpoint_identity(
        {
            "path": str(behavior_path),
            "digest": "sha256:"
            + hashlib.sha256(behavior_path.read_bytes()).hexdigest(),
        }
    )
    chosen, source = train_pure_rl._exact_regression_rollback_identity(
        {
            "matchup_runtime_activation": {
                "receipt": str(tmp_path / "missing.json"),
                "learner_digest": "sha256:" + "0" * 64,
            }
        },
        exact_anchor=anchor,
        behavior_before=behavior,
    )
    assert chosen == behavior
    assert source == "runtime_receipt_invalid_preserve_behavior"


def test_self_play_refill_capacity_is_bounded_and_seed_disjoint() -> None:
    jobs = [
        {
            "job_index": index,
            "seed": 1_000 + index,
            "target_provenance": {"self_play": True},
        }
        for index in range(8)
    ]

    refill = train_pure_rl._self_play_refill_capacity_jobs(jobs, fraction=0.5)

    assert len(refill) == 4
    assert [job["job_index"] for job in refill] == [8, 9, 10, 11]
    assert {job["seed"] for job in refill}.isdisjoint(
        {job["seed"] for job in jobs}
    )
    assert all(
        job["target_provenance"]["replacement_capacity"] is True
        for job in refill
    )
    assert train_pure_rl._self_play_refill_capacity_jobs(
        jobs, fraction=2.0
    )
    assert len(
        train_pure_rl._self_play_refill_capacity_jobs(jobs, fraction=2.0)
    ) == len(jobs)
    globally_disjoint = train_pure_rl._self_play_refill_capacity_jobs(
        jobs,
        fraction=0.5,
        first_job_index=100,
    )
    assert [job["job_index"] for job in globally_disjoint] == [100, 101, 102, 103]


def _collection_result(
    job_index: int,
    *,
    replacement_for: int | None = None,
    failed: bool = False,
) -> dict:
    provenance = {"collect": "self_play", "self_play": True}
    if replacement_for is not None:
        provenance.update(
            replacement_capacity=True,
            replacement_for_job_index=replacement_for,
        )
    return {
        "job_index": job_index,
        "self_play": True,
        "our_failed": failed,
        "record_json": (
            None
            if failed
            else json.dumps(
                {
                    "episode_id": f"episode-{job_index}",
                    "seat": 0,
                    "archetype": "alakazam",
                    "opp_archetype": "alakazam",
                    "deck": [1] * 60,
                    "value": 1.0,
                    "steps": [
                        {
                            "env_step": 0,
                            "action": [1],
                            "observation": {},
                        }
                    ],
                    "target_provenance": provenance,
                }
            )
        ),
    }


def test_spare_results_are_isolated_and_only_replace_missing_primary(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00001.jsonl"
    spool = tmp_path / "iter_00001.replacements.jsonl"
    writer = train_pure_rl.CompactShardWriter(shard)
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
    }
    retained: set[int] = set()
    runtime_rows: list[dict[str, object]] = []
    train_pure_rl._consume_results(
        iter(
            [
                _collection_result(0),
                _collection_result(1, failed=True),
                _collection_result(2, replacement_for=0),
                _collection_result(3, replacement_for=1),
            ]
        ),
        writer,
        runtime_rows,
        stats,
        replacement_spool=spool,
        retained_job_indices=retained,
    )

    assert retained == {0}
    assert writer.n_games == 1
    assert stats["with_record"] == 1
    assert stats["replacement_capacity_staged_source_games"] == 2
    assert [row["job_index"] for row in runtime_rows] == [0]

    promoted = train_pure_rl._promote_replacement_spool(
        spool,
        missing_job_indices={1},
        writer=writer,
    )
    assert promoted["promoted_job_indices"] == {1}
    assert promoted["promoted_source_games"] == 1
    assert len(promoted["promoted_runtime_audit_rows"]) == 1
    promoted_audit = promoted["promoted_runtime_audit_rows"][0]
    assert promoted_audit["job_index"] == 1
    assert promoted_audit["replacement_attempt_job_index"] == 3
    assert promoted_audit["promoted_replacement"] is True
    assert writer.n_games == 2
    assert not spool.exists()
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == ["episode-0", "episode-3"]
    assert rows[1]["target_provenance"]["replacement_for_job_index"] == 1


def test_spare_can_fill_a_contract_equivalent_missing_schedule_cell(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter_00001.jsonl"
    spool = tmp_path / "iter_00001.replacements.jsonl"
    writer = train_pure_rl.CompactShardWriter(shard)
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
    }
    train_pure_rl._consume_results(
        iter([_collection_result(2, replacement_for=0)]),
        writer,
        [],
        stats,
        replacement_spool=spool,
        retained_job_indices=set(),
    )
    primary_jobs = [
        {
            "job_index": index,
            "our_seat": -1,
            "opponent_id": "",
            "archetype": "alakazam",
            "opp_archetype": "alakazam",
            "target_provenance": {"self_play": True},
        }
        for index in (0, 1)
    ]

    promoted = train_pure_rl._promote_replacement_spool(
        spool,
        missing_job_indices={1},
        writer=writer,
        primary_jobs=primary_jobs,
    )

    assert promoted["promoted_job_indices"] == {1}
    assert writer.n_games == 1
    row = json.loads(shard.read_text().strip())
    provenance = row["target_provenance"]
    assert provenance["replacement_for_job_index"] == 1
    assert provenance["replacement_original_for_job_index"] == 0
    audit = promoted["promoted_runtime_audit_rows"][0]
    assert audit["job_index"] == 1
    assert audit["replacement_original_for_job_index"] == 0


def test_explicit_mirror_failover_can_fill_its_original_missing_cell(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "iter.jsonl"
    spool = tmp_path / "iter.replacements.jsonl"
    writer = train_pure_rl.CompactShardWriter(shard)
    current_digest = "sha256:" + "c" * 64
    record = {
        "episode_id": "mirror-retry",
        "seat": 1,
        "archetype": "alakazam",
        "opp_archetype": "alakazam",
        "deck": [1] * 60,
        "value": 1.0,
        "steps": [{"env_step": 0, "action": [1], "observation": {}}],
        "target_provenance": {
            "self_play": True,
            "replacement_capacity": True,
            "replacement_for_job_index": 7,
            "replacement_rescheduled_inapplicable_opponent": True,
            "opponent_checkpoint_digest": current_digest,
        },
    }
    train_pure_rl._append_replacement_spool(
        spool,
        replacement_for_job_index=7,
        records=[record],
        runtime_audit_row={
            "job_index": 101,
            "our_seat": 1,
            "opponent_id": "self:current.pt",
            "archetype": "alakazam",
            "invalid": False,
        },
        schedule_contract={
            "our_seat": 1,
            "opponent_id": "self:current.pt",
            "archetype": "alakazam",
            "opp_archetype": "alakazam",
            "opponent_checkpoint_digest": current_digest,
            "opponent_content_digest": "",
            "opponent_training_group": "",
        },
    )
    original_job = {
        "job_index": 7,
        "our_seat": 1,
        "opponent_id": "self:broken-history.pt",
        "archetype": "alakazam",
        "opp_archetype": "alakazam",
        "opponent_checkpoint_digest": "sha256:" + "b" * 64,
        "target_provenance": {"self_play": True},
    }

    promoted = train_pure_rl._promote_replacement_spool(
        spool,
        missing_job_indices={7},
        writer=writer,
        primary_jobs=[original_job],
    )

    assert promoted["promoted_job_indices"] == {7}
    assert writer.n_games == 1
    audit = promoted["promoted_runtime_audit_rows"][0]
    assert audit["job_index"] == 7
    assert audit["replacement_attempt_job_index"] == 101


def test_spare_result_fails_closed_without_isolated_spool(tmp_path: Path) -> None:
    writer = train_pure_rl.CompactShardWriter(tmp_path / "iter.jsonl")
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
    }
    with pytest.raises(RuntimeError, match="isolated replacement spool"):
        train_pure_rl._consume_results(
            iter([_collection_result(2, replacement_for=0)]),
            writer,
            [],
            stats,
        )


def test_targeted_replacements_preserve_schedule_cell_and_use_disjoint_seed() -> None:
    primary = [
        {
            "job_index": index,
            "seed": 2_700_000 + index,
            "our_seat": index % 2,
            "opponent_id": f"opponent-{index % 3}",
            "target_provenance": {
                "collection_job_index": index,
                "self_play": bool(index < 2),
            },
        }
        for index in range(4)
    ]

    retries = train_pure_rl._targeted_replacement_jobs(
        primary,
        missing_job_indices={1, 3},
        retry_round=2,
        first_job_index=100,
    )

    assert [job["job_index"] for job in retries] == [100, 101]
    assert [job["seed"] for job in retries] == [2_780_001, 2_780_003]
    assert [job["our_seat"] for job in retries] == [1, 1]
    assert [job["opponent_id"] for job in retries] == [
        "opponent-1",
        "opponent-0",
    ]
    assert [
        job["target_provenance"]["replacement_for_job_index"]
        for job in retries
    ] == [1, 3]
    assert all(
        job["target_provenance"]["replacement_round"] == 3
        for job in retries
    )


def test_second_self_play_retry_reschedules_only_to_current_mirror() -> None:
    target = {
        "job_index": 7,
        "seed": 2_700_007,
        "our_seat": 1,
        "checkpoint": "/models/current.pt",
        "checkpoint_digest": "sha256:" + "c" * 64,
        "opponent_checkpoint": "/models/broken-history.pt",
        "opponent_checkpoint_digest": "sha256:" + "b" * 64,
        "opponent_id": "self:broken-history.pt",
        "archetype": "alakazam",
        "opp_archetype": "alakazam",
        "target_provenance": {"self_play": True},
    }

    original_retry = train_pure_rl._targeted_replacement_jobs(
        [target],
        missing_job_indices={7},
        retry_round=0,
        first_job_index=100,
    )[0]
    mirror_retry = train_pure_rl._targeted_replacement_jobs(
        [target],
        missing_job_indices={7},
        retry_round=1,
        first_job_index=101,
    )[0]

    assert original_retry["opponent_checkpoint"] == "/models/broken-history.pt"
    assert (
        original_retry["target_provenance"][
            "replacement_rescheduled_inapplicable_opponent"
        ]
        is False
    )
    assert mirror_retry["opponent_checkpoint"] == "/models/current.pt"
    assert mirror_retry["opponent_checkpoint_digest"] == "sha256:" + "c" * 64
    assert mirror_retry["opponent_id"] == "self:current.pt"
    assert mirror_retry["collect_both_seats"] is True
    provenance = mirror_retry["target_provenance"]
    assert provenance["replacement_for_job_index"] == 7
    assert provenance["replacement_rescheduled_inapplicable_opponent"] is True
    assert provenance["replacement_original_opponent"] == {
        "opponent_id": "self:broken-history.pt",
        "opponent_checkpoint": "/models/broken-history.pt",
        "opponent_checkpoint_digest": "sha256:" + "b" * 64,
    }
def test_targeted_replacement_uses_different_exact_contract_source() -> None:
    primary = [
        {
            "job_index": index,
            "seed": 2_700_000 + index,
            "our_seat": 0,
            "opponent_id": "self:learner.pt",
            "archetype": "alakazam",
            "opp_archetype": "alakazam",
            "checkpoint_digest": "learner",
            "opponent_checkpoint_digest": "learner",
            "our_deck": [1] * 60,
            "opp_deck": [1] * 60,
            "collect_both_seats": True,
            "target_provenance": {
                "collection_job_index": index,
                "self_play": True,
            },
        }
        for index in range(4)
    ]

    [retry] = train_pure_rl._targeted_replacement_jobs(
        primary,
        missing_job_indices={1},
        retry_round=0,
        first_job_index=100,
    )

    assert retry["job_index"] == 100
    assert retry["target_provenance"]["replacement_for_job_index"] == 1
    assert retry["target_provenance"]["collection_job_index"] == 1
    assert retry["target_provenance"]["replacement_retry_source_job_index"] != 1
    assert train_pure_rl._replacement_schedule_contract_from_job(retry) == (
        train_pure_rl._replacement_schedule_contract_from_job(primary[1])
    )
    assert retry["seed"] not in {job["seed"] for job in primary}
from scripts.apply_archetype_label_integrity_at_boundary import (
    load_v17_migration_receipt_chain,
    validate_v17_migration_receipt,
    validate_v17_migration_receipt_chain,
)


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
        assert job["target_provenance"]["collect"] == "formal_eval"
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
    assert all(
        job["target_provenance"]["collect"]
        == (
            "official_target"
            if job["target_provenance"]["opponent_training_group"]
            == "official_target"
            else "public_mix"
        )
        for job in jobs
    )
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


def test_training_collection_rejects_control_ids_digests_and_group_aliases() -> None:
    control_digest = "sha256:" + "a" * 64
    safe_digest = "sha256:" + "b" * 64
    registry = {
        "controls": [
            {
                "opponent_id": "control-canonical",
                "content_digest": control_digest,
            }
        ]
    }
    safe_job = {
        "training_eligible": True,
        "opponent_id": "safe-public",
        "spec": {"content_digest": safe_digest},
        "target_provenance": {
            "opponent_training_group": "diverse_public",
            "opponent_content_digest": safe_digest,
        },
    }
    train_pure_rl._assert_training_jobs_exclude_research_controls(
        [safe_job], registry
    )

    mutations = (
        lambda job: job.update(opponent_id="control-canonical"),
        lambda job: job["spec"].update(content_digest=control_digest),
        lambda job: job["target_provenance"].update(
            opponent_content_digest=control_digest
        ),
        lambda job: job["target_provenance"].update(
            opponent_training_group=train_pure_rl.RESEARCH_CONTROL_GROUP
        ),
        lambda job: job.update(training_eligible=False),
    )
    for mutation in mutations:
        corrupted = json.loads(json.dumps(safe_job))
        mutation(corrupted)
        with pytest.raises(RuntimeError, match="training|research control"):
            train_pure_rl._assert_training_jobs_exclude_research_controls(
                [corrupted], registry
            )


def test_research_and_gate_content_aliases_cannot_enter_public_mix() -> None:
    controls_digest = "sha256:" + "a" * 64
    gate_digest = "sha256:" + "b" * 64
    specs = [
        SimpleNamespace(id="control-canonical"),
        SimpleNamespace(id="control-alias"),
        SimpleNamespace(id="gate-canonical"),
        SimpleNamespace(id="gate-alias"),
        SimpleNamespace(id="safe-public"),
    ]
    safe, aliases = train_pure_rl._exclude_protected_baseline_aliases(
        specs=specs,
        excluded_ids={"control-canonical", "gate-canonical"},
        digest_by_id={
            "control-alias": controls_digest,
            "gate-alias": gate_digest,
            "safe-public": "sha256:" + "c" * 64,
        },
        protected_digests={controls_digest, gate_digest},
    )
    assert [spec.id for spec in safe] == ["safe-public"]
    assert aliases == ["control-alias", "gate-alias"]


def test_collect_wave_runs_research_controls_as_non_replay_measurement_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[tuple[str, list[str]]] = []

    class FakeProgress:
        def __init__(self, *, stage: str, **_kwargs) -> None:
            self.stage = stage

        def close(self) -> None:
            return None

    class FakePool:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def imap_unordered(self, _fn, jobs):
            return iter(jobs)

    def fake_consume(results, _writer, _rows, _stats, *, progress, **_kwargs):
        stages.append(
            (progress.stage, [str(row["opponent_id"]) for row in results])
        )

    monkeypatch.setattr(train_pure_rl, "_TqdmProgress", FakeProgress)
    monkeypatch.setattr(train_pure_rl, "_consume_results", fake_consume)
    monkeypatch.setattr(
        train_pure_rl.StreamingReplayCache,
        "maybe_start",
        lambda *_args, **_kwargs: pytest.fail(
            "diagnostic measurement must not start replay caching"
        ),
    )
    monkeypatch.setattr("poke_bot.worker_pool.WorkerPool", FakePool)

    research = [
        {"opponent_id": "iono"},
        {"opponent_id": "dragapult-ex"},
    ]
    writer, _rows, stats = train_pure_rl._collect_wave(
        self_play_jobs=[],
        baseline_jobs=research,
        shard_path=tmp_path / "iter_00001.jsonl",
        n_workers=1,
        leaf_channel=None,
        remote_farm=None,
        worker_play=lambda row: row,
        worker_self_play=lambda row: row,
        iteration=1,
        stage_label="measure:research_controls",
        replay_eligible=False,
    )

    assert stages == [
        ("measure:research_controls", ["iono", "dragapult-ex"]),
    ]
    assert writer.n_games == writer.n_decisions == 0
    assert stats["n_research_control_jobs"] == 0
    assert stats["n_baseline_jobs"] == 2


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


def test_committed_gate_result_pointer_recovers_from_immutable_commit(
    tmp_path: Path,
) -> None:
    (tmp_path / "commits").mkdir()
    result = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "gate-v2",
        "iteration": 4,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "passed": True,
        "pipeline_gate_passed": True,
    }
    initial = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "r",
        "mode": "specialist",
        "next_iteration": 4,
        "last_completed_iteration": 3,
        "history": [],
    }
    committed = {
        **initial,
        "next_iteration": 5,
        "last_completed_iteration": 4,
        "updated_at_utc": "2026-07-21T12:00:00Z",
        "history": [
            {
                "iteration": 4,
                "completed": True,
                "active_gate_result": result,
            }
        ],
    }
    train_pure_rl._atomic_json(tmp_path / "loop_state.json", initial)
    commit_path = tmp_path / "commits" / "iter_00004.json"
    commit_path.write_text(json.dumps(committed), encoding="utf-8")
    pointer = tmp_path / "state" / "gate-result.json"

    published = train_pure_rl._publish_committed_active_gate_result(
        run_dir=tmp_path,
        active_gate={"id": "gate-v2"},
        result_pointer=pointer,
    )
    assert published == (pointer.resolve(), commit_path.resolve())
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["committed"] is True
    assert payload["commit"] == str(commit_path.resolve())
    assert payload["commit_digest"] == train_pure_rl._canonical_digest(committed)
    assert payload["checkpoint_digest"] == result["checkpoint_digest"]
    assert json.loads((tmp_path / "loop_state.json").read_text()) == committed

    # A mutable pointer can be recreated idempotently, but cannot rewrite the
    # immutable result for the same iteration.
    assert train_pure_rl._publish_committed_active_gate_result(
        run_dir=tmp_path,
        active_gate={"id": "gate-v2"},
        result_pointer=pointer,
    ) == (pointer.resolve(), commit_path.resolve())
    payload["checkpoint_digest"] = "sha256:" + "b" * 64
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicts with immutable commit"):
        train_pure_rl._publish_committed_active_gate_result(
            run_dir=tmp_path,
            active_gate={"id": "gate-v2"},
            result_pointer=pointer,
        )


def test_new_lineage_replaces_global_result_pointer_from_prior_gate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "new-run"
    (run_dir / "commits").mkdir(parents=True)
    result = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "gate-new",
        "iteration": 0,
        "checkpoint_digest": "sha256:" + "d" * 64,
        "passed": False,
        "pipeline_gate_passed": False,
    }
    committed = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "new-run",
        "mode": "specialist",
        "next_iteration": 1,
        "last_completed_iteration": 0,
        "history": [
            {
                "iteration": 0,
                "completed": True,
                "active_gate_result": result,
            }
        ],
    }
    train_pure_rl._atomic_json(run_dir / "loop_state.json", committed)
    commit_path = run_dir / "commits" / "iter_00000.json"
    commit_path.write_text(json.dumps(committed), encoding="utf-8")
    pointer = tmp_path / "global" / "gate-result.json"
    train_pure_rl._atomic_json(
        pointer,
        {
            "schema": "poke_bot.public_agent_gate_result/v1",
            "gate_id": "gate-old",
            "iteration": 20,
            "checkpoint_digest": "sha256:" + "e" * 64,
            "commit": str(tmp_path / "old-run" / "commits" / "iter_00020.json"),
            "committed": True,
        },
    )

    assert train_pure_rl._publish_committed_active_gate_result(
        run_dir=run_dir,
        active_gate={"id": "gate-new"},
        result_pointer=pointer,
    ) == (pointer.resolve(), commit_path.resolve())
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["gate_id"] == "gate-new"
    assert payload["iteration"] == 0
    assert payload["commit"] == str(commit_path.resolve())
    payload["iteration"] = 5
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ahead of commit history"):
        train_pure_rl._publish_committed_active_gate_result(
            run_dir=run_dir,
            active_gate={"id": "gate-new"},
            result_pointer=pointer,
        )


def test_same_lineage_publishes_owner_authorized_lc55_gate_revision(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    roster = [f"opponent-{index}" for index in range(8)]
    result = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "alakazam-strong-public-roster-lc55-v2",
        "iteration": 21,
        "checkpoint_digest": "sha256:" + "f" * 64,
        "matchups": [{"opponent_id": opponent} for opponent in roster],
        "passed": False,
    }
    committed = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "run",
        "mode": "specialist",
        "next_iteration": 22,
        "last_completed_iteration": 21,
        "history": [
            {
                "iteration": 21,
                "completed": True,
                "active_gate_result": result,
            }
        ],
    }
    train_pure_rl._atomic_json(run_dir / "loop_state.json", committed)
    commit_path = run_dir / "commits" / "iter_00021.json"
    commit_path.write_text(json.dumps(committed), encoding="utf-8")
    pointer = tmp_path / "gate-result.json"
    train_pure_rl._atomic_json(
        pointer,
        {
            **result,
            "gate_id": "alakazam-strong-public-roster-v1",
            "iteration": 20,
            "commit": str(run_dir / "commits" / "iter_00020.json"),
            "matchups": [{"opponent_id": opponent} for opponent in roster],
            "committed": True,
        },
    )
    active_gate = {
        "id": "alakazam-strong-public-roster-lc55-v2",
        "roster": [{"opponent_id": opponent} for opponent in roster],
        "evaluation": {"games_total": 2000, "games_per_opponent": 250},
        "pass_criteria": {"skill_weighted_confidence_lower": 0.55},
    }

    assert train_pure_rl._publish_committed_active_gate_result(
        run_dir=run_dir,
        active_gate=active_gate,
        result_pointer=pointer,
    ) == (pointer.resolve(), commit_path.resolve())
    published = json.loads(pointer.read_text(encoding="utf-8"))
    assert published["gate_id"] == "alakazam-strong-public-roster-lc55-v2"
    assert published["iteration"] == 21


def test_same_lineage_publishes_owner_authorized_lc50_after_iter30(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "commits").mkdir(parents=True)
    roster = [f"opponent-{index}" for index in range(8)]
    result = {
        "schema": "poke_bot.public_agent_gate_result/v1",
        "gate_id": "alakazam-strong-public-roster-lc50-after-iter30-v1",
        "iteration": 31,
        "checkpoint_digest": "sha256:" + "f" * 64,
        "matchups": [{"opponent_id": opponent} for opponent in roster],
        "passed": False,
    }
    committed = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "run",
        "mode": "specialist",
        "next_iteration": 32,
        "last_completed_iteration": 31,
        "history": [
            {"iteration": 31, "completed": True, "active_gate_result": result}
        ],
    }
    train_pure_rl._atomic_json(run_dir / "loop_state.json", committed)
    commit_path = run_dir / "commits" / "iter_00031.json"
    commit_path.write_text(json.dumps(committed), encoding="utf-8")
    pointer = tmp_path / "gate-result.json"
    train_pure_rl._atomic_json(
        pointer,
        {
            **result,
            "gate_id": "alakazam-strong-public-roster-lc55-v2",
            "iteration": 30,
            "commit": str(run_dir / "commits" / "iter_00030.json"),
            "matchups": [{"opponent_id": opponent} for opponent in roster],
            "committed": True,
        },
    )
    active_gate = {
        "id": "alakazam-strong-public-roster-lc50-after-iter30-v1",
        "roster": [{"opponent_id": opponent} for opponent in roster],
        "evaluation": {"games_total": 2000, "games_per_opponent": 250},
        "pass_criteria": {"skill_weighted_confidence_lower": 0.50},
        "activation": {
            "schema": "poke_bot.iteration_gate_fallback_activation/v1",
            "prior_gate_id": "alakazam-strong-public-roster-lc55-v2",
            "activate_after_completed_iteration": 30,
            "observed_completed_iteration": 30,
            "prior_gate_passed": False,
            "only_changed_criterion": "skill_weighted_confidence_lower",
            "prior_confidence_lower": 0.55,
            "active_confidence_lower": 0.50,
        },
    }

    assert train_pure_rl._publish_committed_active_gate_result(
        run_dir=run_dir,
        active_gate=active_gate,
        result_pointer=pointer,
    ) == (pointer.resolve(), commit_path.resolve())
    assert json.loads(pointer.read_text())["gate_id"] == active_gate["id"]


def test_lc50_fallback_is_a_terminal_child_of_requested_lc55() -> None:
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "ops/alakazam_gate_program_v1.json")
        .read_text(encoding="utf-8")
    )
    assert train_pure_rl._terminal_gate_target_matches(
        requested_gate_id=contract["active_gate_id"],
        passed_gate_id=contract["fallback_transition"]["id"],
        base_contract=contract,
    )
    assert not train_pure_rl._terminal_gate_target_matches(
        requested_gate_id="alakazam-strong-public-roster-lc55-v2",
        passed_gate_id="unrelated-gate",
        base_contract=contract,
    )


def test_gate_history_pass_check_respects_iteration_boundary() -> None:
    state = {
        "history": [
            {
                "iteration": 30,
                "active_gate_result": {"gate_id": "lc55", "passed": True},
            }
        ]
    }
    assert not train_pure_rl._gate_passed_in_history(
        state, gate_id="lc55", through_iteration=29
    )
    assert train_pure_rl._gate_passed_in_history(
        state, gate_id="lc55", through_iteration=30
    )


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


def test_receipted_orphan_candidate_is_preserved_for_promotion_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("shards", "checkpoints", "metrics", "eval", "commits"):
        (tmp_path / name).mkdir()
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    candidate = tmp_path / "checkpoints" / "iter_00001.pt"
    shard.write_bytes(b"complete shard")
    candidate.write_bytes(b"complete candidate")
    contract = {"source": {"source_tree_sha256": "sha256:test"}}
    contract_digest = train_pure_rl._design_fingerprint(contract)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "design_contract": contract,
                "design_fingerprint": contract_digest,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "next_iteration": 1,
        "learner": {"digest": "sha256:parent"},
    }
    design_fingerprint = "sha256:" + "d" * 64
    receipt = {
        "checkpoint_digest": "sha256:parent",
        "design_fingerprint_at_collection": design_fingerprint,
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    observed = {}

    def verify(path: Path, **kwargs):
        observed.update({"path": path, **kwargs})
        return {"candidate_digest": "sha256:candidate"}

    monkeypatch.setattr(
        train_pure_rl, "_verified_orphan_candidate_result", verify
    )

    assert train_pure_rl._recover_interrupted_iteration(tmp_path, state) is None
    assert shard.is_file()
    assert candidate.is_file()
    assert observed["path"] == candidate.resolve()
    assert observed["parent_digest"] == "sha256:parent"
    assert observed["behavior_digest"] == "sha256:parent"
    assert observed["design_fingerprint"] == (design_fingerprint,)


def test_receipted_candidate_and_research_result_are_preserved_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "shards",
        "checkpoints",
        "metrics",
        "eval",
        "commits",
        "research_controls",
    ):
        (tmp_path / name).mkdir()
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    candidate = tmp_path / "checkpoints" / "iter_00001.pt"
    research = tmp_path / "research_controls" / "iter_00001.json"
    shard.write_bytes(b"complete shard")
    candidate.write_bytes(b"complete candidate")
    candidate_digest = train_pure_rl._sha256_file(candidate)
    research_result = {
        "schema": train_pure_rl.RESEARCH_CONTROL_RESULT_SCHEMA,
        "iteration": 1,
        "checkpoint_digest": candidate_digest,
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "formal_eval": False,
        "action_selection": "greedy",
        "games": 1000,
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "seed_disjoint": True,
            "package_disjoint_from_active_gate": True,
            "replay_records_written": 0,
            "measurement_plan": {"iteration": 1},
        },
        "result_path": str(research.resolve()),
    }
    research.write_text(json.dumps(research_result), encoding="utf-8")
    contract = {"source": {"source_tree_sha256": "sha256:test"}}
    contract_digest = train_pure_rl._design_fingerprint(contract)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "design_contract": contract,
                "design_fingerprint": contract_digest,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "next_iteration": 1,
        "learner": {"digest": "sha256:parent"},
    }
    receipt = {
        "checkpoint_digest": "sha256:parent",
        "design_fingerprint_at_collection": contract_digest,
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_orphan_candidate_result",
        lambda *_args, **_kwargs: {"candidate_digest": candidate_digest},
    )
    assert train_pure_rl._recover_interrupted_iteration(tmp_path, state) is None
    assert shard.is_file()
    assert candidate.is_file()
    assert research.is_file()


def test_unsafe_research_result_is_not_preserved_with_orphan_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "shards",
        "checkpoints",
        "metrics",
        "eval",
        "commits",
        "research_controls",
    ):
        (tmp_path / name).mkdir()
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    candidate = tmp_path / "checkpoints" / "iter_00001.pt"
    research = tmp_path / "research_controls" / "iter_00001.json"
    shard.write_bytes(b"complete shard")
    candidate.write_bytes(b"complete candidate")
    research.write_text(
        json.dumps(
            {
                "schema": train_pure_rl.RESEARCH_CONTROL_RESULT_SCHEMA,
                "iteration": 1,
                "checkpoint_digest": train_pure_rl._sha256_file(candidate),
                "training_eligible": True,
                "audit": {"measurement_plan": {"iteration": 1}},
                "result_path": str(research.resolve()),
            }
        ),
        encoding="utf-8",
    )
    contract = {"source": {"source_tree_sha256": "sha256:test"}}
    contract_digest = train_pure_rl._design_fingerprint(contract)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"design_contract": contract, "design_fingerprint": contract_digest}
        ),
        encoding="utf-8",
    )
    state = {"next_iteration": 1, "learner": {"digest": "sha256:parent"}}
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_receipt",
        lambda *_args, **_kwargs: {
            "checkpoint_digest": "sha256:parent",
            "design_fingerprint_at_collection": contract_digest,
        },
    )
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_orphan_candidate_result",
        lambda *_args, **_kwargs: {
            "candidate_digest": train_pure_rl._sha256_file(candidate)
        },
    )

    failure = train_pure_rl._recover_interrupted_iteration(tmp_path, state)
    assert failure is not None and failure.is_file()
    assert not shard.exists()
    assert not candidate.exists()
    assert not research.exists()


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
    assert receipt["source_games"] == 2
    assert receipt["trajectory_records"] == 2
    assert receipt["shard"]["games"] == 2
    assert receipt["shard"]["decisions"] == 2

    # Recovery preserves exactly the verified shard rather than quarantining it.
    assert train_pure_rl._recover_interrupted_iteration(tmp_path, state) is None
    assert shard.is_file()
    bundle = train_pure_rl._completed_collection_bundle(tmp_path, state, contract)
    assert bundle is not None and bundle["recovered"] is True
    assert bundle["design_fingerprint_at_collection"] == design_fingerprint
    assert bundle["writer"].n_games == 2
    assert bundle["writer"].n_decisions == 2
    behavior = train_pure_rl._collection_behavior_identity(bundle)
    assert behavior.path == str(checkpoint.resolve())
    assert behavior.digest == checkpoint_digest

    # A clean-boundary design change cannot consume a shard collected under
    # the prior opponent mixture, even when its game count/cache still match.
    drifted_contract = json.loads(json.dumps(contract))
    drifted_contract["collection"] = {
        "group_games_per_iteration": {
            "self_play": 0,
            "strong_public_practice": 1,
            "diverse_public": 1,
        },
        "research_control_phase": {
            "games_per_iteration": 1000,
            "training_eligible": False,
            "replay_eligible": False,
        },
    }
    assert (
        train_pure_rl._verified_completed_collection_receipt(
            tmp_path, state, drifted_contract
        )
        is None
    )


def test_completed_collection_keeps_source_games_distinct_from_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "collection_receipts").mkdir()
    checkpoint = tmp_path / "learner.pt"
    checkpoint.write_bytes(b"learner")
    checkpoint_digest = train_pure_rl._sha256_file(checkpoint)
    shard = tmp_path / "iter_00001.jsonl"
    shard.write_text("{}\n{}\n{}\n", encoding="utf-8")
    writer = train_pure_rl.CompactShardWriter.from_completed_shard(
        shard,
        n_games=3,
        n_decisions=7,
        elapsed_sec=1.0,
    )
    contract = {
        "games": {"per_iteration": 2, "minimum_usable_fraction": 0.98},
        "learner": {"max_context": 64},
    }
    state = {
        "next_iteration": 1,
        "design_fingerprint": "sha256:design",
        "learner": {"path": str(checkpoint), "digest": checkpoint_digest},
    }

    monkeypatch.setattr(
        train_pure_rl,
        "validated_replay_cache_manifest",
        lambda path, **_kwargs: {
            "records": 3,
            "sequences": 3,
            "dropped": 0,
            "covered_bytes": path.stat().st_size,
            "manifest_path": str(tmp_path / "cache/manifest.json"),
            "signature": {"source": str(path)},
        },
    )
    receipt = train_pure_rl._commit_completed_collection_receipt(
        run_dir=tmp_path,
        state=state,
        contract=contract,
        iteration=1,
        shard=shard,
        checkpoint=checkpoint,
        checkpoint_digest=checkpoint_digest,
        stats={
            "retained_source_games": 2,
            "retained_trajectories": 3,
            "trajectories_written": 3,
        },
        started_at=shard.stat().st_mtime - 1.0,
        writer=writer,
    )

    assert receipt["requested_games"] == 2
    assert receipt["source_games"] == 2
    assert receipt["trajectory_records"] == 3
    assert receipt["shard"]["games"] == 3


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


def _passing_active_gate(
    candidate,
    *,
    win_rate: float,
    confidence_lower: float,
    games: int,
) -> dict:
    return {
        "passed": True,
        "skill_weighted_wr": win_rate,
        "confidence_lower": confidence_lower,
        "games": games,
        "checkpoint": candidate.path,
        "checkpoint_digest": candidate.digest,
        "checks": {
            "audit": True,
            "skill_weighted_win_rate": True,
            "skill_weighted_confidence_lower": True,
            "s_tier_mean_floor": True,
            "individual_opponent_floor": True,
        },
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "both_seats": True,
        },
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
                "candidate": champion.as_dict(),
                "active_gate_result": _passing_active_gate(
                    champion,
                    win_rate=0.75,
                    confidence_lower=0.71,
                    games=200,
                ),
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


def test_continue_after_gate_preserves_first_committed_pass(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    first_path.write_bytes(b"first pass")
    second_path.write_bytes(b"second pass")
    from poke_bot.promotion import CheckpointIdentity

    first = CheckpointIdentity.from_path(first_path)
    second = CheckpointIdentity.from_path(second_path)
    first_row = {
        "iteration": 3,
        "completed": True,
        "candidate": first.as_dict(),
        "active_gate_result": _passing_active_gate(
            first,
            win_rate=0.75,
            confidence_lower=0.71,
            games=2000,
        ),
    }
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "continued",
        "mode": "specialist",
        "champion": first.as_dict(),
        "history": [first_row],
    }
    marker = train_pure_rl._ensure_terminal_gate_marker(
        tmp_path, state, preserve_first=True
    )
    first_payload = json.loads(marker.read_text(encoding="utf-8"))

    # A later failed iteration must not hide or replace the archive boundary.
    state["champion"] = second.as_dict()
    state["history"].append(
        {
            "iteration": 4,
            "completed": True,
            "candidate": second.as_dict(),
            "active_gate_result": {
                **_passing_active_gate(
                    second,
                    win_rate=0.49,
                    confidence_lower=0.47,
                    games=2000,
                ),
                "passed": False,
            },
        }
    )
    assert (
        train_pure_rl._ensure_terminal_gate_marker(
            tmp_path, state, preserve_first=True
        )
        == marker
    )
    assert json.loads(marker.read_text(encoding="utf-8")) == first_payload

    # Nor may a later pass overwrite the exact first checkpoint identity.
    state["history"][-1]["active_gate_result"].update(
        {
            "passed": True,
            "skill_weighted_wr": 0.76,
            "confidence_lower": 0.72,
        }
    )
    assert (
        train_pure_rl._ensure_terminal_gate_marker(
            tmp_path, state, preserve_first=True
        )
        == marker
    )
    assert json.loads(marker.read_text(encoding="utf-8")) == first_payload


def test_current_protocol_marker_does_not_replace_historical_marker(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"current protocol pass")
    from poke_bot.promotion import CheckpointIdentity

    candidate = CheckpointIdentity.from_path(checkpoint_path)
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "current-protocol",
        "mode": "specialist",
        "champion": candidate.as_dict(),
        "history": [
            {
                "iteration": 30,
                "completed": True,
                "candidate": candidate.as_dict(),
                "active_gate_result": _passing_active_gate(
                    candidate,
                    win_rate=0.61,
                    confidence_lower=0.56,
                    games=2000,
                ),
            }
        ],
    }
    historical = tmp_path / "SPECIALIST_GATE_PASSED"
    historical.write_text('{"iteration": 20}\n', encoding="utf-8")
    current_name = "SPECIALIST_GATE_PASSED.alakazam-lc55-v2"

    current = train_pure_rl._ensure_terminal_gate_marker(
        tmp_path,
        state,
        preserve_first=True,
        marker_name=current_name,
    )

    assert current == tmp_path / current_name
    assert json.loads(current.read_text(encoding="utf-8"))["iteration"] == 30
    assert historical.read_text(encoding="utf-8") == '{"iteration": 20}\n'
    with pytest.raises(ValueError, match="safe filename"):
        train_pure_rl._ensure_terminal_gate_marker(
            tmp_path, state, marker_name="../unsafe"
        )


def test_continue_after_gate_rejects_uncommitted_marker(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"candidate")
    from poke_bot.promotion import CheckpointIdentity

    candidate = CheckpointIdentity.from_path(checkpoint_path)
    state = {
        "version": train_pure_rl.LOOP_STATE_VERSION,
        "run_name": "continued",
        "mode": "specialist",
        "champion": candidate.as_dict(),
        "history": [
            {
                "iteration": 5,
                "completed": True,
                "candidate": candidate.as_dict(),
                "stage_gate": {
                    "passed": False,
                    "win_rate": 0.49,
                    "confidence_lower": 0.47,
                    "games": 2000,
                },
            }
        ],
    }
    (tmp_path / "SPECIALIST_GATE_PASSED").write_text(
        json.dumps(
            {
                "iteration": 4,
                "wr": 0.8,
                "confidence_lower": 0.7,
                "games": 2000,
                "checkpoint": str(checkpoint_path),
                "checkpoint_digest": candidate.digest,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="absent from committed history"):
        train_pure_rl._ensure_terminal_gate_marker(
            tmp_path, state, preserve_first=True
        )


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


def test_initial_empty_run_allows_receipted_source_migration(
    tmp_path: Path,
) -> None:
    stored = {
        "learner": {
            "games_per_batch": 96,
            "max_decisions_per_batch": 8192,
        },
        "source": {"loader_contract": "sha256:old"},
        "run": {"iterations": 16},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 0,
        "last_completed_iteration": -1,
        "history": [],
    }
    manifest = {
        "design_fingerprint": stored_digest,
        "design_contract": stored,
    }
    current = {
        "learner": {
            "games_per_batch": 96,
            "max_decisions_per_batch": 8192,
        },
        "source": {"loader_contract": "sha256:new"},
        "run": {"iterations": 16},
    }

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="receipt_backed_completed_collection_resume_v1",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    assert state["design_fingerprint"] == digest
    assert len(list((tmp_path / "design_migrations").glob("*.json"))) == 1


def test_initial_collection_resume_preserves_verified_iteration_zero_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = {
        "learner": {
            "games_per_batch": 96,
            "max_decisions_per_batch": 8192,
        },
        "source": {"source_tree_sha256": "sha256:old"},
        "run": {"iterations": 16},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 0,
        "last_completed_iteration": -1,
        "history": [],
    }
    manifest = {
        "design_fingerprint": stored_digest,
        "design_contract": stored,
    }
    (tmp_path / "shards").mkdir()
    shard = tmp_path / "shards" / "iter_00000.jsonl"
    shard.write_text("{}\n", encoding="utf-8")
    receipt = {
        "iteration": 0,
        "checkpoint_digest": "sha256:behavior",
        "design_fingerprint_at_collection": stored_digest,
        "receipt_path": str(
            tmp_path / "collection_receipts" / "iter_00000.json"
        ),
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_across_design_chain",
        lambda *_args, **_kwargs: (receipt, {}),
    )
    current = json.loads(json.dumps(stored))
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="receipt_backed_completed_collection_resume_v1",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    assert shard.is_file()
    migration = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert migration["preserved_completed_collection"]["iteration"] == 0


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


def test_clean_boundary_migration_allows_exact_zero_safe_decision_fusion(
    tmp_path: Path,
) -> None:
    stored = {
        "learner": {
            "alakazam_guide_loss_weight": 0.0,
            "alakazam_guide_targets_enabled": False,
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
            "profile": {"expanded_heads_enabled": True},
        },
        "source": {"source_tree_sha256": "sha256:old"},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 14,
        "last_completed_iteration": 13,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00013.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = json.loads(json.dumps(stored))
    current["learner"].update(
        current_deck_guide_archetype=None,
        current_deck_guide_loss_weight=0.0,
        current_deck_guide_targets_enabled=False,
        current_deck_guide_version=None,
    )
    current["learner"]["profile"].update(
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_width=16,
    )
    current["source"]["source_tree_sha256"] = "sha256:fusion"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="receipt_backed_decision_fusion_warmup_v1",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert set(receipt["changed_paths"]) == {
        *train_pure_rl._DECISION_FUSION_WARMUP_MIGRATION_PATHS,
        "source.source_tree_sha256",
    }


def test_decision_fusion_warmup_composes_with_allowed_operational_change() -> None:
    stored = {
        "learner": {
            "games_per_batch": 120,
            "alakazam_guide_loss_weight": 0.0,
            "alakazam_guide_targets_enabled": False,
            "profile": {"expanded_heads_enabled": True},
        }
    }
    current = json.loads(json.dumps(stored))
    current["learner"].update(
        games_per_batch=240,
        current_deck_guide_archetype=None,
        current_deck_guide_loss_weight=0.0,
        current_deck_guide_targets_enabled=False,
        current_deck_guide_version=None,
    )
    current["learner"]["profile"].update(
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_width=16,
    )
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))

    assert train_pure_rl._safe_decision_fusion_warmup_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_decision_fusion_warmup_v1",
    )

    current["learner"]["epochs"] = 2
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))
    assert not train_pure_rl._safe_decision_fusion_warmup_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_decision_fusion_warmup_v1",
    )


def test_decision_fusion_runtime_activation_is_exact_and_fail_closed() -> None:
    stored = {
        "learner": {
            "profile": {
                "expanded_heads_enabled": True,
                "decision_fusion_enabled": True,
                "decision_fusion_runtime_enabled": False,
                "decision_fusion_width": 16,
            }
        }
    }
    current = json.loads(json.dumps(stored))
    current["learner"]["profile"]["decision_fusion_runtime_enabled"] = True
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))
    assert train_pure_rl._safe_decision_fusion_runtime_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_decision_fusion_runtime_v1",
    )

    current["learner"]["profile"]["decision_fusion_width"] = 32
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))
    assert not train_pure_rl._safe_decision_fusion_runtime_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_decision_fusion_runtime_v1",
    )


def test_current_deck_guide_weight_migration_is_bounded_and_exact() -> None:
    stored = {
        "learner": {
            "alakazam_guide_loss_weight": 0.05,
            "current_deck_guide_loss_weight": 0.05,
            "current_deck_guide_archetype": "teal-mask-ogerpon-ex",
            "current_deck_guide_version": "slop-box-v3",
            "current_deck_guide_targets_enabled": True,
        },
        "expert_rehearsal": {
            "loss_weights": {"alakazam_guide": 0.05},
        },
    }
    current = json.loads(json.dumps(stored))
    current["learner"]["alakazam_guide_loss_weight"] = 0.25
    current["learner"]["current_deck_guide_loss_weight"] = 0.25
    current["expert_rehearsal"]["loss_weights"]["alakazam_guide"] = 0.25
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))

    assert train_pure_rl._safe_current_deck_guide_weight_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_current_deck_guide_weight_curve_v1",
    )

    too_large = json.loads(json.dumps(current))
    too_large["learner"]["alakazam_guide_loss_weight"] = 0.75
    too_large["learner"]["current_deck_guide_loss_weight"] = 0.75
    too_large["expert_rehearsal"]["loss_weights"]["alakazam_guide"] = 0.75
    assert not train_pure_rl._safe_current_deck_guide_weight_migration(
        stored=stored,
        current=too_large,
        changed=sorted(train_pure_rl._changed_design_paths(stored, too_large)),
        reason="receipt_backed_current_deck_guide_weight_curve_v1",
    )
    assert not train_pure_rl._safe_current_deck_guide_weight_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason="unreceipted_weight_change",
    )


def test_teal_tactical_outcome_rebalance_is_bounded_and_exact() -> None:
    stored = {
        "learner": {
            "current_deck_guide_archetype": "teal-mask-ogerpon-ex",
            "current_deck_guide_loss_weight": 0.05,
        }
    }
    current = json.loads(json.dumps(stored))
    current["learner"]["expanded_head_loss_weight_overrides"] = {
        "tactical_outcome": 0.01
    }
    changed = sorted(train_pure_rl._changed_design_paths(stored, current))

    assert train_pure_rl._safe_teal_auxiliary_head_rebalance(
        stored=stored,
        current=current,
        changed=changed,
        reason="receipt_backed_teal_auxiliary_head_rebalance_v1",
    )

    wrong_weight = json.loads(json.dumps(current))
    wrong_weight["learner"]["expanded_head_loss_weight_overrides"][
        "tactical_outcome"
    ] = 0.025
    assert not train_pure_rl._safe_teal_auxiliary_head_rebalance(
        stored=stored,
        current=wrong_weight,
        changed=sorted(
            train_pure_rl._changed_design_paths(stored, wrong_weight)
        ),
        reason="receipt_backed_teal_auxiliary_head_rebalance_v1",
    )
    assert not train_pure_rl._safe_teal_auxiliary_head_rebalance(
        stored=stored,
        current=current,
        changed=changed,
        reason="unreceipted_weight_change",
    )


def test_completed_collection_survives_zero_safe_fusion_warmup(
    tmp_path: Path,
) -> None:
    parent = "sha256:" + "1" * 64
    warmup = "sha256:" + "2" * 64
    material_path = tmp_path / "materialization.json"
    material_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.causal_decision_fusion_checkpoint_migration/v1",
                "parent_checkpoint_digest": parent,
                "migrated_checkpoint_digest": warmup,
                "decision_fusion": {"runtime_enabled": False},
                "proof": {
                    "legacy_tensors_bit_identical": True,
                    "optimizer_existing_state_preserved": True,
                    "zero_safe_initialization": True,
                },
            }
        ),
        encoding="utf-8",
    )
    boundary_path = tmp_path / "boundary.json"
    boundary_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.causal_decision_fusion_boundary_warmup/v1",
                "run_dir": str(tmp_path),
                "boundary": {"next_iteration": 15},
                "parent_learner": {"digest": parent},
                "warmup_learner": {"digest": warmup},
                "materialization_receipt": {
                    "path": str(material_path),
                    "digest": train_pure_rl._sha256_file(material_path),
                },
            }
        ),
        encoding="utf-8",
    )
    state = {
        "next_iteration": 15,
        "learner": {"digest": warmup},
        "decision_fusion_activation": {
            "schema": "poke_bot.causal_decision_fusion_boundary_warmup/v1",
            "phase": "training_warmup",
            "boundary_next_iteration": 15,
            "learner_digest": warmup,
            "runtime_enabled": False,
            "serving_eligible": False,
            "receipt": str(boundary_path),
            "receipt_digest": train_pure_rl._sha256_file(boundary_path),
        },
    }
    assert train_pure_rl._completed_collection_checkpoint_matches_state(
        run_dir=tmp_path,
        state=state,
        collection_digest=parent,
    )

    material = json.loads(material_path.read_text(encoding="utf-8"))
    material["proof"]["legacy_tensors_bit_identical"] = False
    material_path.write_text(json.dumps(material), encoding="utf-8")
    assert not train_pure_rl._completed_collection_checkpoint_matches_state(
        run_dir=tmp_path,
        state=state,
        collection_digest=parent,
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda current: current["learner"]["profile"].update(
                decision_fusion_runtime_enabled=True
            ),
            "receipt_backed_decision_fusion_warmup_v1",
        ),
        (
            lambda current: current["learner"].update(
                current_deck_guide_loss_weight=0.1
            ),
            "receipt_backed_decision_fusion_warmup_v1",
        ),
        (lambda _current: None, "generic migration"),
    ],
)
def test_clean_boundary_decision_fusion_migration_fails_closed(
    tmp_path: Path,
    mutation: Any,
    reason: str,
) -> None:
    stored = {
        "learner": {
            "alakazam_guide_loss_weight": 0.0,
            "alakazam_guide_targets_enabled": False,
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
            "profile": {"expanded_heads_enabled": True},
        }
    }
    current = json.loads(json.dumps(stored))
    current["learner"].update(
        current_deck_guide_archetype=None,
        current_deck_guide_loss_weight=0.0,
        current_deck_guide_targets_enabled=False,
        current_deck_guide_version=None,
    )
    current["learner"]["profile"].update(
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_width=16,
    )
    mutation(current)
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 14,
        "last_completed_iteration": 13,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00013.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="non-operational fields"):
        train_pure_rl._validate_or_migrate_design_fingerprint(
            run_dir=tmp_path,
            state=state,
            manifest=manifest,
            current=current,
            allow_clean_boundary_migration=True,
            migration_reason=reason,
        )


def test_clean_boundary_migration_allows_only_safe_ceiling_reduction(
    tmp_path: Path,
) -> None:
    stored = {
        "run": {"iterations": 100},
        "learner": {
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
        },
        "collection": {
            "strong_public_practice": {
                "seed_contract": {"iterations": 100}
            }
        },
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 6,
        "last_completed_iteration": 5,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00005.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = json.loads(json.dumps(stored))
    current["run"]["iterations"] = 36
    current["collection"]["strong_public_practice"]["seed_contract"][
        "iterations"
    ] = 36

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="enforce specialist ceiling 35",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert receipt["changed_paths"] == [
        "collection.strong_public_practice.seed_contract.iterations",
        "run.iterations",
    ]


@pytest.mark.parametrize("iterations", [6, 101])
def test_clean_boundary_migration_rejects_unsafe_ceiling_change(
    tmp_path: Path,
    iterations: int,
) -> None:
    stored = {"run": {"iterations": 100}}
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 6,
        "last_completed_iteration": 5,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00005.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = {"run": {"iterations": iterations}}

    with pytest.raises(RuntimeError, match="non-operational fields"):
        train_pure_rl._validate_or_migrate_design_fingerprint(
            run_dir=tmp_path,
            state=state,
            manifest=manifest,
            current=current,
            allow_clean_boundary_migration=True,
            migration_reason="unsafe ceiling change",
        )


def test_source_only_migration_preserves_verified_completed_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "iter_00001.jsonl").write_text("{}\n", encoding="utf-8")
    receipt = {
        "iteration": 1,
        "checkpoint_digest": "sha256:behavior",
        "design_fingerprint_at_collection": stored_digest,
        "receipt_path": str(tmp_path / "collection_receipts" / "iter_00001.json"),
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_across_design_chain",
        lambda *_args, **_kwargs: (receipt, {}),
    )
    current = json.loads(json.dumps(stored))
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="resume receipt-verified collection after source-only fix",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    migration = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert migration["preserved_completed_collection"]["iteration"] == 1

    # A second source-only correction at the same interrupted boundary must
    # carry the already verified collection through the prior migration.  The
    # collection was generated under the original fingerprint, so requiring it
    # to equal only the immediately previous fingerprint would strand a valid
    # append-only recovery after two independent code fixes.
    current_again = json.loads(json.dumps(current))
    current_again["source"]["source_tree_sha256"] = "sha256:newer"
    digest_again = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current_again,
        allow_clean_boundary_migration=True,
        migration_reason="second source-only fix over the same collection",
    )

    assert digest_again == train_pure_rl._design_fingerprint(current_again)
    migrations = sorted(
        (tmp_path / "design_migrations").glob("migration_*.json")
    )
    assert len(migrations) == 2
    carried = json.loads(migrations[-1].read_text())
    assert carried["preserved_completed_collection"]["iteration"] == 1


def test_source_only_fix_preserves_collection_after_quarantined_fusion_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = {
        "learner": {"games_per_batch": 96, "max_decisions_per_batch": 8192},
        "source": {"source_tree_sha256": "sha256:original", "git_head": None},
    }
    warmup = json.loads(json.dumps(original))
    warmup["learner"]["profile"] = {
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": False,
        "decision_fusion_width": 16,
    }
    current = json.loads(json.dumps(warmup))
    current["source"]["source_tree_sha256"] = "sha256:source-fix"
    original_digest = train_pure_rl._design_fingerprint(original)
    warmup_digest = train_pure_rl._design_fingerprint(warmup)
    state = {
        "design_fingerprint": warmup_digest,
        "next_iteration": 15,
        "last_completed_iteration": 14,
    }
    manifest = {
        "design_fingerprint": original_digest,
        "design_contract": original,
    }
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00014.json").write_text(
        json.dumps(
            {
                "next_iteration": 15,
                "design_fingerprint": original_digest,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards/iter_00015.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "design_migrations").mkdir()
    (tmp_path / "design_migrations/migration_0001.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "receipt": str(
                    tmp_path / "design_migrations/migration_0001.json"
                ),
                "boundary_next_iteration": 15,
                "previous_fingerprint": original_digest,
                "current_fingerprint": warmup_digest,
                "previous_contract": original,
                "current_contract": warmup,
                # Deliberately absent: the shard was quarantined during the
                # warmup migration and therefore could not be recorded here.
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "iteration": 15,
        "checkpoint_digest": "sha256:behavior-parent",
        "design_fingerprint_at_collection": original_digest,
        "receipt_path": str(
            tmp_path / "collection_receipts/iter_00015.json"
        ),
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_across_design_chain",
        lambda *_args, **_kwargs: (receipt, original),
    )
    monkeypatch.setattr(
        train_pure_rl,
        "_completed_collection_checkpoint_matches_state",
        lambda **_kwargs: True,
    )

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="source-only fix after zero-safe fusion warmup",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    migrations = sorted(
        (tmp_path / "design_migrations").glob("migration_*.json")
    )
    assert len(migrations) == 2
    carried = json.loads(migrations[-1].read_text())
    assert carried["preserved_completed_collection"]["iteration"] == 15


def test_source_only_migration_preserves_candidate_and_research_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "learner": {"digest": "sha256:parent"},
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    for name in ("commits", "shards", "checkpoints", "research_controls"):
        (tmp_path / name).mkdir()
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps({**state, "next_iteration": 1}), encoding="utf-8"
    )
    shard = tmp_path / "shards" / "iter_00001.jsonl"
    candidate = tmp_path / "checkpoints" / "iter_00001.pt"
    research = tmp_path / "research_controls" / "iter_00001.json"
    shard.write_text("{}\n", encoding="utf-8")
    candidate.write_bytes(b"candidate")
    candidate_digest = train_pure_rl._sha256_file(candidate)
    research.write_text(
        json.dumps(
            {
                "schema": train_pure_rl.RESEARCH_CONTROL_RESULT_SCHEMA,
                "iteration": 1,
                "checkpoint_digest": candidate_digest,
                "training_eligible": False,
                "replay_eligible": False,
                "diagnostic_only": True,
                "included_in_gate_pass": False,
                "gate_weight": 0.0,
                "formal_eval": False,
                "action_selection": "greedy",
                "games": 1000,
                "audit": {
                    "passed": True,
                    "exact_distribution": True,
                    "exact_weights": True,
                    "seed_disjoint": True,
                    "package_disjoint_from_active_gate": True,
                    "replay_records_written": 0,
                    "measurement_plan": {"iteration": 1},
                },
                "result_path": str(research.resolve()),
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "iteration": 1,
        "checkpoint_digest": "sha256:behavior",
        "design_fingerprint_at_collection": stored_digest,
        "receipt_path": str(tmp_path / "collection_receipts" / "iter_00001.json"),
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_across_design_chain",
        lambda *_args, **_kwargs: (receipt, {}),
    )
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_orphan_candidate_result",
        lambda *_args, **_kwargs: {"candidate_digest": candidate_digest},
    )
    current = json.loads(json.dumps(stored))
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="preserve completed candidate research transaction",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    assert shard.is_file() and candidate.is_file() and research.is_file()
    migration = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert migration["preserved_completed_collection"]["iteration"] == 1


def test_completed_collection_does_not_mask_non_source_design_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits" / "iter_00000.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "iter_00001.jsonl").write_text("{}\n", encoding="utf-8")
    receipt = {
        "iteration": 1,
        "checkpoint_digest": "sha256:behavior",
        "design_fingerprint_at_collection": stored_digest,
    }
    monkeypatch.setattr(
        train_pure_rl,
        "_verified_completed_collection_across_design_chain",
        lambda *_args, **_kwargs: (receipt, {}),
    )
    current = json.loads(json.dumps(stored))
    current["source"]["source_tree_sha256"] = "sha256:new"
    current["learner"]["games_per_batch"] = 192

    with pytest.raises(RuntimeError, match="clean boundary"):
        train_pure_rl._validate_or_migrate_design_fingerprint(
            run_dir=tmp_path,
            state=state,
            manifest=manifest,
            current=current,
            allow_clean_boundary_migration=True,
            migration_reason="must remain fail closed",
        )


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


def test_clean_boundary_migration_allows_versioned_expert_contract() -> None:
    assert (
        "expert_rehearsal.rolling_manifest_pointer"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )
    assert (
        "expert_rehearsal.loss_weights"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )
    assert (
        "expert_rehearsal.matchup_adapters"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )
    assert (
        "learner.profile.matchup_adapters_enabled"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )


def test_clean_boundary_migration_allows_explicit_opponent_scope_schema() -> None:
    assert (
        "collection.training_opponent_scope"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )
    assert (
        "collection.external_agents_training_eligible"
        in train_pure_rl._BOUNDARY_MIGRATABLE_DESIGN_PATHS
    )


def test_clean_boundary_migration_allows_only_pinned_hidden_engine_bytes(
    tmp_path: Path,
) -> None:
    stored = {
        "learner": {
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
        },
        "collection": {
            "auxiliary_targets": {
                "hidden_engine": {
                    "path": "/outputs/engines/libcg_hidden_inzi_v1.so",
                    "digest": "sha256:" + "a" * 64,
                    "size": 100,
                }
            }
        },
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 11,
        "last_completed_iteration": 10,
    }
    manifest = {"design_fingerprint": stored_digest, "design_contract": stored}
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00010.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current = json.loads(json.dumps(stored))
    current["collection"]["auxiliary_targets"]["hidden_engine"].update(
        digest="sha256:" + "b" * 64,
        size=200,
    )
    current["source"]["source_tree_sha256"] = "sha256:new"

    digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="restore the audited pinned engine binary",
    )

    assert digest == train_pure_rl._design_fingerprint(current)
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert receipt["changed_paths"] == [
        "collection.auxiliary_targets.hidden_engine.digest",
        "collection.auxiliary_targets.hidden_engine.size",
        "source.source_tree_sha256",
    ]

    changed_path = json.loads(json.dumps(stored))
    changed_path["collection"]["auxiliary_targets"]["hidden_engine"]["path"] = (
        "/tmp/untrusted-engine.so"
    )
    with pytest.raises(RuntimeError, match="non-operational fields"):
        train_pure_rl._validate_or_migrate_design_fingerprint(
            run_dir=tmp_path / "path-change",
            state={
                "design_fingerprint": stored_digest,
                "next_iteration": 11,
                "last_completed_iteration": 10,
            },
            manifest=manifest,
            current=changed_path,
            allow_clean_boundary_migration=True,
            migration_reason="must not move the pinned engine path",
        )


def test_real_v15_effective_to_v17_contract_migration_is_fully_audited(
    tmp_path: Path,
) -> None:
    v15_root = Path(
        "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v15-strong-practice"
    )
    v17_root = Path(
        "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v17-research-controls"
    )
    control_ids = (
        "iono",
        "dragapult-ex",
        "mega-abomasnow-ex",
        "mega-lucario-ex",
    )
    control_specs = [{"id": opponent_id} for opponent_id in control_ids]
    stored = {
        "learner": {
            "games_per_batch": 240,
            "max_decisions_per_batch": 8192,
        },
        "games": {"per_iteration": 8192, "heldout": 2000},
        "gates": {
            "active_contract": {
                "kind": "file",
                "path": str(v15_root / "ops/alakazam_gate_program_v1.json"),
                "digest": "sha256:" + "a" * 64,
                "size": 9713,
            }
        },
        "expert_rehearsal": {
            "rolling_manifest_pointer": "/data/expert.json",
        },
        "collection": {
            "auxiliary_targets": {
                "hidden_engine": {
                    "path": "/home/inzi/poke-bot-agent/outputs/engines/libcg_hidden_inzi_v1.so",
                    "digest": "sha256:" + "c" * 64,
                    "size": 100,
                }
            },
            "behavior_policy": (
                "continuous_learner_after_h2h_safety_and_exact_audit_v2"
            ),
            "strong_public_practice": {
                "enabled": True,
                "fraction_of_public": 0.5,
                "training_eligible": True,
                "formal_eval": False,
            },
        },
        "opponents": {
            "collect": [{"id": "diverse-public"}],
            "heldout": [{"id": "strong-public"}],
            "official_target_training": [{"id": "strong-public"}],
        },
        "source": {"source_tree_sha256": "sha256:old", "git_head": None},
    }
    current = json.loads(json.dumps(stored))
    current["gates"]["active_contract"]["path"] = str(
        v17_root / "ops/alakazam_gate_program_v1.json"
    )
    current["expert_rehearsal"]["loss_weights"] = {
        "archetype": 0.05,
        "opponent_hand": 0.05,
        "opponent_hidden_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "alakazam_guide": 0.05,
    }
    current["collection"]["group_games_per_iteration"] = {
        "self_play": 1024,
        "strong_public_practice": 4584,
        "diverse_public": 2584,
    }
    current["collection"]["auxiliary_targets"]["hidden_engine"].update(
        digest="sha256:" + "d" * 64,
        size=200,
    )
    current["collection"]["strong_public_practice"] = {
        "enabled": True,
        "configured_fraction_of_public": 0.5,
        "effective_fraction_of_public": 4584 / 7168,
        "reclaimed_control_slots": 1000,
        "training_eligible": True,
        "formal_eval": False,
    }
    current["collection"]["research_control_phase"] = {
        "enabled": True,
        "stage": "measure:research_controls",
        "source": "research_control_registry",
        "games_per_iteration": 1000,
        "games_per_control": 250,
        "seat0_games_per_control": 125,
        "seat1_games_per_control": 125,
        "action_selection": "greedy",
        "sampled_behavior_policy": False,
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "additive_to_training_budget": True,
        "formal_eval": False,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "seed_namespace": "eval/research-controls-fixed-manifest-v1",
        "separate_result_artifact": True,
        "roster": control_specs,
        "registry": {
            "kind": "file",
            "path": str(v17_root / "ops/research_control_registry_v1.json"),
            "digest": "sha256:" + "b" * 64,
            "size": 4096,
        },
    }
    current["opponents"]["research_controls"] = control_specs
    current["source"]["source_tree_sha256"] = "sha256:new"

    stored_digest = train_pure_rl._design_fingerprint(stored)
    state = {
        "design_fingerprint": stored_digest,
        "next_iteration": 10,
        "last_completed_iteration": 9,
    }
    manifest = {
        "design_fingerprint": stored_digest,
        "design_contract": stored,
    }
    (tmp_path / "commits").mkdir()
    (tmp_path / "commits/iter_00009.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    current_digest = train_pure_rl._validate_or_migrate_design_fingerprint(
        run_dir=tmp_path,
        state=state,
        manifest=manifest,
        current=current,
        allow_clean_boundary_migration=True,
        migration_reason="canonical-archetype-labels-and-research-controls-v17",
    )
    receipt = json.loads(
        next((tmp_path / "design_migrations").glob("migration_*.json")).read_text()
    )
    assert current_digest == train_pure_rl._design_fingerprint(current)
    assert receipt["current_contract"] == current
    assert receipt["current_fingerprint"] == current_digest
    changed = validate_v17_migration_receipt(
        train_pure_rl,
        receipt,
        target_next_iteration=10,
        staged_root=v17_root,
    )
    assert {
        "collection.group_games_per_iteration",
        "collection.auxiliary_targets.hidden_engine.digest",
        "collection.auxiliary_targets.hidden_engine.size",
        "collection.research_control_phase",
        "expert_rehearsal.loss_weights",
        "gates.active_contract.path",
        "opponents.research_controls",
    }.issubset(changed)

    post_patch = json.loads(json.dumps(current))
    post_patch["source"]["source_tree_sha256"] = "sha256:post-patch"
    post_patch_receipt = {
        "schema": 1,
        "reason": "canonical-archetype-labels-and-research-controls-v17",
        "boundary_next_iteration": 10,
        "changed_paths": ["source.source_tree_sha256"],
        "previous_contract": current,
        "current_contract": post_patch,
        "previous_fingerprint": train_pure_rl._design_fingerprint(current),
        "current_fingerprint": train_pure_rl._design_fingerprint(post_patch),
    }
    chained = validate_v17_migration_receipt_chain(
        train_pure_rl,
        [receipt, post_patch_receipt],
        target_next_iteration=10,
        staged_root=v17_root,
    )
    assert "source.source_tree_sha256" in chained
    post_patch_path = tmp_path / "design_migrations" / "migration_0002.json"
    post_patch_path.write_text(json.dumps(post_patch_receipt), encoding="utf-8")
    chain_paths, loaded_chain = load_v17_migration_receipt_chain(
        tmp_path,
        latest_receipt=post_patch_path,
        reason="canonical-archetype-labels-and-research-controls-v17",
        target_next_iteration=10,
    )
    assert chain_paths == [
        tmp_path / "design_migrations" / "migration_0001.json",
        post_patch_path,
    ]
    assert loaded_chain == [receipt, post_patch_receipt]
    with pytest.raises(RuntimeError, match="missing_required"):
        validate_v17_migration_receipt(
            train_pure_rl,
            post_patch_receipt,
            target_next_iteration=10,
            staged_root=v17_root,
        )

    broken_chain = json.loads(json.dumps(post_patch_receipt))
    broken_chain["previous_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="receipt chain is invalid"):
        validate_v17_migration_receipt_chain(
            train_pure_rl,
            [receipt, broken_chain],
            target_next_iteration=10,
            staged_root=v17_root,
        )

    tampered = json.loads(json.dumps(receipt))
    tampered["current_contract"]["collection"]["research_control_phase"][
        "training_eligible"
    ] = True
    tampered["current_fingerprint"] = train_pure_rl._design_fingerprint(
        tampered["current_contract"]
    )
    with pytest.raises(RuntimeError, match="not the exact v17 design"):
        validate_v17_migration_receipt(
            train_pure_rl,
            tampered,
            target_next_iteration=10,
            staged_root=v17_root,
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


def test_checkpoint_contract_accepts_legacy_disabled_fusion_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy inference opponent may omit only disabled/default fusion fields."""

    from poke_bot import checkpoint, features

    path = tmp_path / "legacy-champion.pt"
    path.write_bytes(b"checkpoint bytes")
    model_config = train_pure_rl.model_config_dict(
        train_pure_rl.pure_rl_model_config()
    )
    for key in (
        "decision_fusion_enabled",
        "decision_fusion_runtime_enabled",
        "decision_fusion_width",
    ):
        model_config.pop(key)
    payload = {
        "model_config": model_config,
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

    assert contract["model_profile"]["decision_fusion_enabled"] is False
    assert contract["model_profile"]["decision_fusion_runtime_enabled"] is False
    assert contract["model_profile"]["decision_fusion_width"] == 16


def test_checkpoint_contract_allows_only_declared_legacy_inference_role_during_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen flat incumbent may coexist with, but never impersonate, the learner."""

    from poke_bot import checkpoint, features

    path = tmp_path / "legacy-inference-champion.pt"
    path.write_bytes(b"checkpoint bytes")
    legacy_config = train_pure_rl.model_config_dict(
        train_pure_rl.pure_rl_model_config()
    )
    legacy_config["decision_fusion_enabled"] = False
    legacy_config["decision_fusion_runtime_enabled"] = False
    payload = {
        "model_config": legacy_config,
        "model_state_dict": {"policy.weight": torch.zeros(3, 4)},
        "provenance": {"feature_schema": features.FEATURE_SCHEMA_VERSION},
        "extra": {"pure_rl": True, "smoke": False},
    }
    expected_cfg = train_pure_rl.pure_rl_model_config()
    expected_cfg.decision_fusion_enabled = True
    expected_cfg.decision_fusion_runtime_enabled = False
    monkeypatch.setattr(
        train_pure_rl,
        "pure_rl_model_config",
        lambda **_kwargs: expected_cfg,
    )
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

    with pytest.raises(RuntimeError, match="decision_fusion_enabled"):
        train_pure_rl._checkpoint_contract(path, smoke=False)

    contract = train_pure_rl._checkpoint_contract(
        path,
        smoke=False,
        allow_legacy_inference_profile=True,
    )
    assert contract["legacy_inference_profile"] is True
    assert contract["model_profile"]["decision_fusion_enabled"] is True
    assert contract["model_profile"]["decision_fusion_runtime_enabled"] is False

    payload["model_config"]["dropout"] = 0.25
    with pytest.raises(RuntimeError, match="dropout"):
        train_pure_rl._checkpoint_contract(
            path,
            smoke=False,
            allow_legacy_inference_profile=True,
        )


def test_checkpoint_contract_rejects_enabled_fusion_when_profile_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_bot import checkpoint, features

    path = tmp_path / "wrong-role.pt"
    path.write_bytes(b"checkpoint bytes")
    model_config = train_pure_rl.model_config_dict(
        train_pure_rl.pure_rl_model_config()
    )
    model_config["decision_fusion_enabled"] = True
    payload = {
        "model_config": model_config,
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

    with pytest.raises(RuntimeError, match="decision_fusion_enabled"):
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
