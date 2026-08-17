"""Tests for archetype-generic RTP pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.recursive_turn_planner.pipeline import (
    ArchetypeRTPJob,
    _normalize_sha256,
    _plan_r197_batch_cap,
    _sha256_file,
    _select_r197_episodes,
    _verify_expected_digest,
    example_registry_jobs,
    load_archetype_registry,
    run_archetype_rtp_pipeline,
    run_registry,
    select_jobs,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_example_registry_has_core_archetypes() -> None:
    jobs = example_registry_jobs()
    ids = {j.specialist_id for j in jobs}
    assert "alakazam" in ids
    assert "marnie-s-grimmsnarl-ex" in ids
    assert "crustle" in ids


@pytest.mark.unit
def test_load_example_yaml_registry() -> None:
    path = ROOT / "config" / "rtp_archetype_pipeline.example.yaml"
    jobs = load_archetype_registry(path)
    assert len(jobs) >= 3
    assert all(j.specialist_id for j in jobs)


@pytest.mark.unit
def test_select_jobs_filters() -> None:
    jobs = example_registry_jobs()
    only = select_jobs(jobs, specialist_ids=["Alakazam", "crustle"])
    assert [j.specialist_id for j in only] == ["alakazam", "crustle"]
    ready = select_jobs(jobs, only_ready=True)
    assert ready == []  # example paths empty


@pytest.mark.unit
def test_supplied_parent_and_shard_digests_are_byte_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"rtp provenance")
    actual = _sha256_file(artifact)
    assert _normalize_sha256(actual.upper(), field="digest") == actual
    assert _verify_expected_digest(
        expected=actual,
        actual=actual,
        field="parent_digest",
    )
    with pytest.raises(ValueError, match="parent_digest mismatch"):
        _verify_expected_digest(
            expected="sha256:" + "0" * 64,
            actual=actual,
            field="parent_digest",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        _normalize_sha256("not-a-digest", field="training_shard_digest")


@pytest.mark.unit
def test_r197_job_is_explicit_while_legacy_job_remains_configurable() -> None:
    # Legacy callers are not silently constrained by the r197 candidate.
    legacy = ArchetypeRTPJob(
        specialist_id="legacy",
        profile="pure_rl",
        num_plan_candidates=2,
        max_recursion_depth=0,
        max_neural_passes=1,
    )
    assert legacy.profile == "pure_rl"

    with pytest.raises(ValueError, match="requires parent_checkpoint"):
        ArchetypeRTPJob(
            specialist_id="alakazam",
            parent_digest="sha256:" + "a" * 64,
            complete_action_corpus="/immutable/r197-corpus",
            complete_action_corpus_manifest_digest="sha256:" + "b" * 64,
            complete_action_corpus_receipt_digest="sha256:" + "d" * 64,
            complete_action_corpus_source_pointer_digest="sha256:" + "c" * 64,
            complete_action_corpus_selection_plan_digest="sha256:" + "e" * 64,
            complete_action_corpus_train_selection_digest="sha256:" + "f" * 64,
            complete_action_corpus_heldout_selection_digest="sha256:" + "1" * 64,
            profile="pure_rl_r197",
            max_neural_passes=256,
            max_runtime_action_combos=1024,
            split_seed=5_000_000,
            max_train_games=512,
            max_heldout_games=128,
            max_train_batches=32_000,
            max_heldout_batches=8_000,
        )

    r197 = ArchetypeRTPJob(
        specialist_id="alakazam",
        parent_checkpoint="/immutable/r195.pt",
        parent_digest="sha256:" + "a" * 64,
        complete_action_corpus="/immutable/r197-corpus",
        complete_action_corpus_manifest_digest="sha256:" + "b" * 64,
        complete_action_corpus_receipt_digest="sha256:" + "d" * 64,
        complete_action_corpus_source_pointer_digest="sha256:" + "c" * 64,
        complete_action_corpus_selection_plan_digest="sha256:" + "e" * 64,
        complete_action_corpus_train_selection_digest="sha256:" + "f" * 64,
        complete_action_corpus_heldout_selection_digest="sha256:" + "1" * 64,
        profile="pure_rl_r197",
        d_model=96,
        num_plan_candidates=4,
        max_recursion_depth=2,
        max_neural_passes=256,
        max_runtime_action_combos=1024,
        split_seed=5_000_000,
        max_train_games=512,
        max_heldout_games=128,
        max_train_batches=32_000,
        max_heldout_batches=8_000,
    )
    assert r197.ready_for_host_train is True
    with pytest.raises(ValueError, match="max_runtime_action_combos=1024"):
        ArchetypeRTPJob(
            specialist_id="alakazam",
            parent_checkpoint="/immutable/r195.pt",
            parent_digest="sha256:" + "a" * 64,
            complete_action_corpus="/immutable/r197-corpus",
            complete_action_corpus_manifest_digest="sha256:" + "b" * 64,
            complete_action_corpus_receipt_digest="sha256:" + "d" * 64,
            complete_action_corpus_source_pointer_digest="sha256:" + "c" * 64,
            complete_action_corpus_selection_plan_digest="sha256:" + "e" * 64,
            complete_action_corpus_train_selection_digest="sha256:" + "f" * 64,
            complete_action_corpus_heldout_selection_digest="sha256:" + "1" * 64,
            profile="pure_rl_r197",
            max_neural_passes=256,
            max_runtime_action_combos=256,
            split_seed=5_000_000,
            max_train_games=512,
            max_heldout_games=128,
            max_train_batches=32_000,
            max_heldout_batches=8_000,
        )
    with pytest.raises(ValueError, match="max_neural_passes=256"):
        ArchetypeRTPJob(
            specialist_id="alakazam",
            parent_checkpoint="/immutable/r195.pt",
            parent_digest="sha256:" + "a" * 64,
            complete_action_corpus="/immutable/r197-corpus",
            complete_action_corpus_manifest_digest="sha256:" + "b" * 64,
            complete_action_corpus_receipt_digest="sha256:" + "d" * 64,
            complete_action_corpus_source_pointer_digest="sha256:" + "c" * 64,
            complete_action_corpus_selection_plan_digest="sha256:" + "e" * 64,
            complete_action_corpus_train_selection_digest="sha256:" + "f" * 64,
            complete_action_corpus_heldout_selection_digest="sha256:" + "1" * 64,
            profile="pure_rl_r197",
            max_neural_passes=4,
            split_seed=5_000_000,
            max_train_games=512,
            max_heldout_games=128,
            max_train_batches=32_000,
            max_heldout_batches=8_000,
        )
    with pytest.raises(ValueError, match="global authorized ceiling"):
        ArchetypeRTPJob(
            specialist_id="legacy",
            profile="pure_rl",
            max_neural_passes=257,
        )


@pytest.mark.unit
def test_r197_selection_and_batch_caps_keep_whole_episodes() -> None:
    rows = [
        {"episode_id": episode_id}
        for episode_id, n_rows in (("a", 3), ("b", 2), ("c", 4), ("d", 1))
        for _ in range(n_rows)
    ]
    selected, selection = _select_r197_episodes(
        rows,
        split="train",
        limit=3,
        selection_seed=197,
    )
    selected_again, selection_again = _select_r197_episodes(
        reversed(rows),
        split="train",
        limit=3,
        selection_seed=197,
    )
    assert selected == selected_again
    assert selection["selection_sha256"] == selection_again["selection_sha256"]

    counts = {"a": 3, "b": 2, "c": 4, "d": 1}
    retained, cap = _plan_r197_batch_cap(
        counts,
        episode_order=selected,
        batch_cap=4,
        split="train",
    )
    assert set(retained).issubset(set(selected))
    assert cap["retained_batch_count_pre_encoding"] <= 4
    assert cap["row_level_sampling"] is False


@pytest.mark.unit
def test_run_single_and_fleet_synthetic(tmp_path: Path) -> None:
    job = ArchetypeRTPJob(
        specialist_id="alakazam",
        d_model=96,
        epochs=1,
        also_poke_rlm=True,
        seed=1,
    )
    result = run_archetype_rtp_pipeline(
        job, out_root=tmp_path / "fleet", synthetic=True, n_synthetic=12
    )
    assert result.specialist_id == "alakazam"
    assert Path(result.rtp_checkpoint).is_file()
    assert Path(result.poke_rlm_checkpoint).is_file()
    assert (tmp_path / "fleet" / "alakazam" / "pipeline_summary.json").is_file()

    fleet = run_registry(
        [
            ArchetypeRTPJob(specialist_id="marnie-s-grimmsnarl-ex", epochs=1, seed=2),
            ArchetypeRTPJob(specialist_id="crustle", epochs=1, seed=3),
        ],
        out_root=tmp_path / "fleet2",
        synthetic=True,
        n_synthetic=8,
    )
    assert fleet["n_ok"] == 2
    assert fleet["n_errors"] == 0
    assert (tmp_path / "fleet2" / "fleet_summary.json").is_file()


@pytest.mark.unit
def test_cli_list_and_smoke(tmp_path: Path) -> None:
    import subprocess
    import sys

    reg = ROOT / "config" / "rtp_archetype_pipeline.example.yaml"
    list_proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_rtp_archetype_pipeline.py"),
            "--registry",
            str(reg),
            "--out-dir",
            str(tmp_path / "x"),
            "--list",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert list_proc.returncode == 0, list_proc.stderr
    payload = json.loads(list_proc.stdout)
    assert payload["n"] >= 3

    run_proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_rtp_archetype_pipeline.py"),
            "--registry",
            str(reg),
            "--out-dir",
            str(tmp_path / "run"),
            "--specialist",
            "alakazam",
            "--synthetic",
            "--n-synthetic",
            "8",
            "--epochs",
            "1",
            "--also-poke-rlm",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, run_proc.stderr
    assert (tmp_path / "run" / "alakazam" / "pipeline_summary.json").is_file()
