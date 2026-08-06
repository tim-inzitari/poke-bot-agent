"""Tests for archetype-generic RTP pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.recursive_turn_planner.pipeline import (
    ArchetypeRTPJob,
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
