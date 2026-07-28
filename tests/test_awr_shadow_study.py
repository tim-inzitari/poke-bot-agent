from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot import features
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage
from poke_bot.pure_rl.awr_shadow_study import (
    BETA_VALUES,
    FIXED_DECISION_CAP,
    FIXED_GRAD_CLIP,
    FIXED_LR,
    FIXED_WEIGHT_DECAY,
    exact_training_order_contract,
    assert_shadow_gpu_idle,
    file_digest,
    matched_sequential_evaluation,
    prepare_beta_manifest,
    prepare_stage2_manifest,
    read_json,
    verify_frozen_inputs,
)
from poke_bot.pure_rl.dataset_bridge import _status_path
from poke_bot.train import (
    BatchMetrics,
    _merge_metrics,
    _set_exact_awr_weight_quantiles,
)


def _write_compact(path: Path, *, first: int, games: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(games):
        index = first + offset
        rows.append(
            {
                "episode_id": f"episode-{index}",
                "seat": index % 2,
                "archetype": "alakazam",
                "opp_archetype": "iono",
                "deck": [1] * 60,
                "value": 1.0 if index % 2 else -1.0,
                "decisions": [
                    {
                        "env_step": 0,
                        "selected_index": 0,
                        "n_options": 1,
                        "action": [0],
                        "observation": {},
                    }
                ],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_collection_receipt(shard: Path, receipt: Path, iteration: int) -> None:
    rows = [json.loads(line) for line in shard.read_text().splitlines() if line]
    decisions = sum(len(row["decisions"]) for row in rows)
    payload = {
        "schema": "poke_bot.completed_collection/v1",
        "iteration": iteration,
        "requested_games": len(rows),
        "shard": {
            "path": str(shard.resolve()),
            "sha256": file_digest(shard),
            "size": shard.stat().st_size,
            "games": len(rows),
            "decisions": decisions,
        },
        "replay_cache": {
            "covered_bytes": shard.stat().st_size,
            "records": len(rows),
            "sequences": len(rows),
            "dropped": 0,
            "signature": {"max_context": 320},
        },
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload) + "\n")


def _prepared(tmp_path: Path) -> tuple[Path, dict]:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"immutable-checkpoint")
    baseline = tmp_path / "baseline-contract.json"
    baseline.write_text('{"official":"v1"}\n')
    shard0 = tmp_path / "live" / "shards" / "iter_00001.jsonl"
    shard1 = tmp_path / "live" / "shards" / "iter_00002.jsonl"
    _write_compact(shard0, first=0)
    _write_compact(shard1, first=2)
    receipt0 = tmp_path / "live" / "collection_receipts" / "iter_00001.json"
    receipt1 = tmp_path / "live" / "collection_receipts" / "iter_00002.json"
    _write_collection_receipt(shard0, receipt0, 1)
    _write_collection_receipt(shard1, receipt1, 2)
    manifest_path = prepare_beta_manifest(
        output_dir=tmp_path / "shadow",
        parent_checkpoint=parent,
        replay_shards=[shard0, shard1],
        replay_receipts=[receipt0, receipt1],
        baseline_contract=baseline,
        shadow_device="cuda:0",
        production_device="cuda:1",
        seed=7,
    )
    return manifest_path, read_json(manifest_path)


def test_beta_manifest_freezes_exact_controls_and_cannot_promote(tmp_path: Path) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    controls = manifest["controls"]
    assert controls["optimizer"] == "AdamW"
    assert controls["learning_rate"] == FIXED_LR
    assert controls["weight_decay"] == FIXED_WEIGHT_DECAY
    assert controls["gradient_clip_norm"] == FIXED_GRAD_CLIP
    assert controls["max_decisions_per_batch"] == FIXED_DECISION_CAP
    assert [row["awr_beta"] for row in manifest["beta_variants"]] == list(
        BETA_VALUES
    )
    invariants = {
        (
            row["parent_checkpoint_sha256"],
            row["replay_set_sha256"],
            row["split_seed"],
            row["batch_order_seed"],
            row["controls_sha256"],
        )
        for row in manifest["beta_variants"]
    }
    assert len(invariants) == 1
    assert manifest["promotion_disabled"] is True
    assert manifest["promotion_enabled"] is False
    assert manifest["auto_apply"] is False
    assert manifest["production_mutation_allowed"] is False
    assert manifest["service_control_allowed"] is False
    assert manifest["next_stage"]["status"] == "blocked_until_final_beta_report"
    assert all(
        proposal["enabled"] is False
        for proposal in manifest["research_proposals"].values()
    )
    receipt = read_json(manifest_path.parent / "study_manifest.receipt.json")
    assert receipt["artifact_sha256"] == file_digest(manifest_path)


def test_frozen_input_verification_detects_replay_tamper(tmp_path: Path) -> None:
    _manifest_path, manifest = _prepared(tmp_path)
    verify_frozen_inputs(manifest)
    shard = Path(manifest["replay"]["shards"][0]["path"])
    shard.chmod(0o644)
    shard.write_text(shard.read_text() + "{}\n")
    with pytest.raises(RuntimeError, match="replay invariant changed"):
        verify_frozen_inputs(manifest)


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    value = features.SparseVector()
    for index in range(words):
        value.word_start()
        value.add((offset + index) % 32, 1.0)
    return value


def _decision(index: int) -> DecisionSample:
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=_sparse(2, index + 1),
        action=[index % 2],
        action_combo_index=index % 2,
        action_combos=[[0], [1]],
        env_step=index,
        action_token=_sparse(1, index + 2),
        policy_stages=[
            PolicyStage(
                options=_sparse(2, index + 1),
                action_combos=[[0], [1]],
                target_index=index % 2,
            )
        ],
    )


def test_exact_training_order_contract_is_repeatable() -> None:
    dataset = BootstrapDataset(
        sequences=[
            GameSequence(
                episode_id=f"g{game}",
                seat=game % 2,
                archetype="alakazam",
                opp_archetype="iono",
                deck=[1] * 60,
                value=1.0,
                decisions=[_decision(game), _decision(game + 10)],
            )
            for game in range(8)
        ]
    )
    controls = {
        "max_context": 320,
        "val_frac": 0.25,
        "split_seed": 11,
        "batch_order_seed": 11,
        "epochs": 2,
        "games_per_batch": 3,
        "max_decisions_per_batch": 8,
    }
    first = exact_training_order_contract(dataset, controls)
    second = exact_training_order_contract(dataset, controls)
    assert first == second
    assert first["train_unique_game_sequences"] == 6
    assert first["validation_unique_game_sequences"] == 2
    assert first["optimizer_decision_exposures"] == 24
    changed = exact_training_order_contract(
        dataset, {**controls, "batch_order_seed": 12}
    )
    assert changed["split_sha256"] == first["split_sha256"]
    assert changed["batch_order_sha256"] != first["batch_order_sha256"]


def test_matched_evaluation_requires_every_variant_seed_opponent_and_seat(
    tmp_path: Path,
) -> None:
    _manifest_path, manifest = _prepared(tmp_path)
    receipts = []
    strengths = {}
    for index, variant in enumerate(manifest["beta_variants"]):
        variant_id = variant["id"]
        digest = f"sha256:{index + 1:064x}"
        receipts.append(
            {
                "variant_id": variant_id,
                "awr_beta": variant["awr_beta"],
                "result": {"candidate": {"path": f"/{variant_id}.pt", "sha256": digest}},
            }
        )
        strengths[variant_id] = (0.40, 0.80, 0.60)[index]

    rows = []
    schedule = [
        *manifest["evaluation"]["official_jobs"],
        *manifest["evaluation"]["parent_jobs"],
    ]
    for receipt in receipts:
        variant_id = receipt["variant_id"]
        strength = strengths[variant_id]
        for sequence, expected in enumerate(schedule):
            # Deterministic exact proportions while retaining all schedule keys.
            score = 1.0 if (sequence % 10) < int(strength * 10) else 0.0
            row = {
                **expected,
                "variant_id": variant_id,
                "candidate_sha256": receipt["result"]["candidate"]["sha256"],
                "score": score,
            }
            if expected["target_type"] == "current_checkpoint":
                row["target_sha256"] = manifest["parent_checkpoint"]["sha256"]
            rows.append(row)
    report = matched_sequential_evaluation(manifest, receipts, rows)
    assert report["matched_complete"] is True
    assert report["pairing_claimed"] is False
    assert report["selected_variant"] == manifest["beta_variants"][1]["id"]
    assert report["per_interval_alpha"] < 0.05

    broken = rows[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        matched_sequential_evaluation(manifest, receipts, broken)


def test_stage2_cannot_prepare_from_training_only_report(tmp_path: Path) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    training_only = tmp_path / "training-only.json"
    training_only.write_text(
        json.dumps(
            {
                "status": "training_complete_evaluation_required",
                "study_manifest_sha256": file_digest(manifest_path),
                "next_stage_unlocked": False,
            }
        )
    )
    shards = []
    receipts = []
    for index in range(4):
        shard = tmp_path / "stage2-source" / f"iter_{index:05d}.jsonl"
        _write_compact(shard, first=100 + index * 2)
        shards.append(shard)
        receipt = tmp_path / "stage2-receipts" / f"iter_{index:05d}.json"
        _write_collection_receipt(shard, receipt, index)
        receipts.append(receipt)
    with pytest.raises(ValueError, match="remains blocked"):
        prepare_stage2_manifest(
            beta_manifest_path=manifest_path,
            final_beta_report_path=training_only,
            output_dir=tmp_path / "stage2",
            replay_shards=shards,
            replay_receipts=receipts,
            seed=19,
        )
    assert manifest["next_stage"]["auto_start"] is False


def test_stage2_matrix_is_guarded_and_reports_unique_exposure_counts(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    selected = manifest["beta_variants"][1]
    candidate = tmp_path / "selected-beta.pt"
    candidate.write_bytes(b"selected-beta-candidate")
    candidate_digest = file_digest(candidate)
    final_report = tmp_path / "beta_report.final.json"
    final_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "next_stage_unlocked": True,
                "study_manifest_sha256": file_digest(manifest_path),
                "selected_variant": selected["id"],
                "training_variants": [
                    {
                        "variant_id": selected["id"],
                        "awr_beta": selected["awr_beta"],
                        "result": {
                            "candidate": {
                                "path": str(candidate),
                                "sha256": candidate_digest,
                            }
                        },
                    }
                ],
            }
        )
    )
    shards = []
    replay_receipts = []
    for index in range(4):
        shard = tmp_path / "stage2-source" / f"iter_{index:05d}.jsonl"
        _write_compact(shard, first=200 + index * 2)
        shards.append(shard)
        receipt = tmp_path / "stage2-receipts" / f"iter_{index:05d}.json"
        _write_collection_receipt(shard, receipt, index)
        replay_receipts.append(receipt)
    stage2_path = prepare_stage2_manifest(
        beta_manifest_path=manifest_path,
        final_beta_report_path=final_report,
        output_dir=tmp_path / "stage2",
        replay_shards=shards,
        replay_receipts=replay_receipts,
        seed=23,
    )
    stage2 = read_json(stage2_path)
    assert len(stage2["matrix"]) == 8
    assert stage2["fixed_controls"]["max_decisions_per_batch"] == 8192
    assert stage2["fixed_controls"]["max_context"] == 320
    assert stage2["promotion_disabled"] is True
    assert stage2["auto_apply"] is False
    four = stage2["replay_profiles"]["4_shards_x_1_epoch"]
    two = stage2["replay_profiles"]["2_shards_x_2_epochs"]
    assert four["unique_game_sequences"] == 8
    assert two["unique_game_sequences"] == 4
    assert four["decision_exposures"] == two["decision_exposures"] == 8


def test_awr_metric_merge_preserves_observed_max_and_effective_fraction() -> None:
    merged = _merge_metrics(
        [
            BatchMetrics(
                n_decisions=2,
                awr_weight_mean=1.5,
                awr_weight_sum=3.0,
                awr_weight_sq_sum=5.0,
                awr_weight_p50=1.0,
                awr_weight_p95=2.0,
                awr_weight_max_observed=2.0,
            ),
            BatchMetrics(
                n_decisions=2,
                awr_weight_mean=5.5,
                awr_weight_sum=11.0,
                awr_weight_sq_sum=61.0,
                awr_weight_p50=1.0,
                awr_weight_p95=10.0,
                awr_weight_max_observed=10.0,
            ),
        ]
    )
    assert merged.awr_weight_max_observed == 10.0
    assert 0.0 < merged.awr_effective_sample_fraction <= 1.0
    exact = _set_exact_awr_weight_quantiles(
        merged, [10.0, 1.0, 2.0, 3.0, 4.0]
    )
    assert exact.awr_weight_p50 == 3.0
    assert exact.awr_weight_p95 == 4.0
    assert exact.awr_weight_max_observed == 10.0


def test_shadow_loader_can_suppress_live_run_status_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "live" / "shards" / "iter_00001.jsonl"
    _write_compact(shard, first=0)
    monkeypatch.setenv("POKEBOT_REPLAY_STATUS_DISABLED", "1")
    assert _status_path(shard) is None


def test_shadow_gpu_guard_rejects_foreign_compute_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def idle(command, **_kwargs):
        if any("--query-gpu=" in token for token in command):
            return "0, GPU-3080\n1, GPU-BLACKWELL\n"
        return ""

    monkeypatch.setattr("subprocess.check_output", idle)
    assert_shadow_gpu_idle("cuda:0")

    def occupied(command, **_kwargs):
        if any("--query-gpu=" in token for token in command):
            return "0, GPU-3080\n1, GPU-BLACKWELL\n"
        return "GPU-3080, 999999, python, 1024\n"

    monkeypatch.setattr("subprocess.check_output", occupied)
    with pytest.raises(RuntimeError, match="not isolated"):
        assert_shadow_gpu_idle("cuda:0")
