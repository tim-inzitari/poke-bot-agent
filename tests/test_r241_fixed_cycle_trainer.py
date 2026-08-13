from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from poke_bot.pure_rl.expert_rehearsal import rehearsal_due
from scripts import train_pure_rl


def _r241_args(*extra: str):
    return train_pure_rl._parse_args(
        [
            "--run-name",
            "alakazam-r241-fixed-cycle-test",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "alakazam",
            "--iterations",
            "10",
            "--fixed-cycle-updates",
            "10",
            "--r241-peak-r195-preservation-receipt",
            "/tmp/r241-peak-r195-preservation-v6.json",
            "--r241-peak-r195-preservation-receipt-sha256",
            "sha256:" + "a" * 64,
            "--expert-rehearsal-every",
            "5",
            "--expert-rehearsal-epochs",
            "5",
            "--terminal-expert-rehearsal",
            *extra,
        ]
    )


def _r274_args(*extra: str):
    return train_pure_rl._parse_args(
        [
            "--run-name",
            "alakazam-r274-fixed-cycle-test",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "alakazam",
            "--iterations",
            "25",
            "--fixed-cycle-updates",
            "25",
            "--r274-submission-boundary-dir",
            "/tmp/r274-submission-boundaries",
            "--r274-r195-research-baseline-receipt",
            "/tmp/r274-r195-research-baseline.json",
            "--r274-r195-research-baseline-receipt-sha256",
            "sha256:" + "c" * 64,
            "--r280-contiguous-expert-pack",
            "/tmp/r279-contiguous-pack.pt",
            "--r280-contiguous-expert-pack-receipt",
            "/tmp/r279-contiguous-pack-receipt.json",
            "--r241-peak-r195-preservation-receipt",
            "/tmp/r241-peak-r195-preservation-v6.json",
            "--r241-peak-r195-preservation-receipt-sha256",
            "sha256:" + "a" * 64,
            "--expert-rehearsal-before-first",
            "--expert-rehearsal-one-time-before",
            "0",
            "--expert-rehearsal-one-time-epochs",
            "25",
            "--expert-rehearsal-every",
            "5",
            "--expert-rehearsal-epochs",
            "5",
            "--terminal-expert-rehearsal",
            *extra,
        ]
    )


def _completed_r241_state(run_dir: Path) -> dict:
    commits = run_dir / "commits"
    commits.mkdir(parents=True)
    history = [
        {"iteration": iteration, "completed": True}
        for iteration in range(10)
    ]
    for iteration in range(10):
        (commits / f"iter_{iteration:05d}.json").write_text(
            json.dumps(
                {
                    "last_completed_iteration": iteration,
                    "next_iteration": iteration + 1,
                }
            ),
            encoding="utf-8",
        )
    refresh_dir = run_dir / "rehearsals"
    refresh_dir.mkdir()
    (refresh_dir / "before_iter_00005.json").write_text(
        json.dumps({"before_iteration": 5, "epochs": 5}),
        encoding="utf-8",
    )
    return {
        "last_completed_iteration": 9,
        "next_iteration": 10,
        "history": history,
    }


def test_r241_fixed_cycle_cli_requires_the_exact_two_refresh_boundaries() -> None:
    args = _r241_args()

    assert train_pure_rl._validate_fixed_cycle_configuration(args) == 10
    assert [
        iteration for iteration in range(11) if rehearsal_due(iteration, 5)
    ] == [5, 10]

    with pytest.raises(SystemExit):
        _r241_args("--expert-rehearsal-epochs", "4")


def test_r274_fixed_cycle_requires_bootstrap_and_four_precollection_refreshes() -> None:
    args = _r274_args()

    assert train_pure_rl._validate_fixed_cycle_configuration(args) == 25
    assert [
        iteration
        for iteration in range(26)
        if train_pure_rl._r241_precollection_refresh_due(
            iteration, {"status": "passed"}, 25
        )
    ] == [5, 10, 15, 20]

    with pytest.raises(SystemExit):
        _r274_args("--expert-rehearsal-one-time-epochs", "24")
    with pytest.raises(SystemExit):
        _r274_args("--no-expert-rehearsal-before-first")


def test_r274_fixed_cycle_accepts_one_receipted_external_bootstrap() -> None:
    args = _r274_args(
        "--no-expert-rehearsal-before-first",
        "--expert-rehearsal-one-time-before",
        "-1",
        "--expert-rehearsal-one-time-epochs",
        "0",
        "--r274-bootstrap-handoff-receipt",
        "/tmp/r274-handoff.json",
        "--r274-bootstrap-handoff-receipt-sha256",
        "sha256:" + "b" * 64,
    )
    assert train_pure_rl._validate_fixed_cycle_configuration(args) == 25

    with pytest.raises(SystemExit):
        _r274_args(
            "--r274-bootstrap-handoff-receipt",
            "/tmp/r274-handoff.json",
            "--r274-bootstrap-handoff-receipt-sha256",
            "sha256:" + "b" * 64,
        )


def test_r274_fixed_cycle_requires_submission_boundary_exchange() -> None:
    args = _r274_args()
    args.r274_submission_boundary_dir = None
    with pytest.raises(ValueError, match="submission-boundary-dir"):
        train_pure_rl._validate_fixed_cycle_configuration(args)


def test_r274_submission_request_and_upload_are_exactly_bound(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "refresh.pt"
    rehearsal = tmp_path / "before_iter_00005.json"
    checkpoint.write_bytes(b"refreshed checkpoint")
    rehearsal.write_text('{"status":"passed"}\n', encoding="utf-8")
    checkpoint_identity = train_pure_rl._r241_file_identity(checkpoint)
    rehearsal_identity = train_pure_rl._r241_file_identity(rehearsal)
    request_path, request = train_pure_rl._materialize_r274_submission_request(
        root=tmp_path / "exchange",
        boundary_update=5,
        checkpoint_identity=checkpoint_identity,
        rehearsal_receipt_identity=rehearsal_identity,
        owner_contract_sha256="sha256:" + "c" * 64,
    )
    assert request_path.is_file()
    assert train_pure_rl._materialize_r274_submission_request(
        root=tmp_path / "exchange",
        boundary_update=5,
        checkpoint_identity=checkpoint_identity,
        rehearsal_receipt_identity=rehearsal_identity,
        owner_contract_sha256="sha256:" + "c" * 64,
    )[1] == request

    upload = {
        "schema": train_pure_rl.R274_SUBMISSION_UPLOAD_SCHEMA,
        "status": "submitted",
        "candidate_id": train_pure_rl.R274_CANDIDATE_ID,
        "boundary_update": 5,
        "request_sha256": request["request_sha256"],
        "checkpoint": checkpoint_identity,
        "direct_policy": True,
        "rtp_enabled": False,
        "remote_submission_id": 55379999,
    }
    upload["receipt_sha256"] = train_pure_rl._canonical_digest(upload)
    assert train_pure_rl._validate_r274_submission_upload(
        upload=upload,
        request=request,
    )["remote_submission_id"] == 55379999

    drifted = json.loads(json.dumps(upload))
    drifted["checkpoint"]["digest"] = "sha256:" + "d" * 64
    drifted["receipt_sha256"] = train_pure_rl._canonical_digest(
        {key: value for key, value in drifted.items() if key != "receipt_sha256"}
    )
    with pytest.raises(RuntimeError, match="does not match"):
        train_pure_rl._validate_r274_submission_upload(
            upload=drifted,
            request=request,
        )


def test_r274_diverse_research_floor_is_128_with_64_each_seat() -> None:
    class Spec:
        def __init__(self, identity: str):
            self.id = identity

    research = Spec(train_pure_rl.R274_R195_RESEARCH_OPPONENT_ID)
    diverse = [research, *(Spec(f"diverse-{index}") for index in range(11))]
    minimums = {str(spec.id): 0 for spec in diverse}
    minimums[research.id] = 128
    schedule = train_pure_rl._interleaved_opponent_schedule(
        7_172,
        priority_specs=[Spec("marnie")],
        diverse_specs=diverse,
        priority_frac=4_586 / 7_172,
        seed=274,
        iteration=0,
        priority_minimum_games={"marnie": 1_024},
        diverse_minimum_games=minimums,
        priority_group="strong_public_practice",
    )
    selected = [row for row, group in schedule if row.id == research.id]
    assert len(schedule) == 7_172
    assert len(selected) >= 128


def test_r274_research_receipt_accepts_only_bound_retained_runtime_rows() -> None:
    digest = "sha256:" + "d" * 64
    rows = [
        {
            "opponent_id": train_pure_rl.R274_R195_RESEARCH_OPPONENT_ID,
            "our_seat": index % 2,
            "opponent_content_digest": digest,
            "opponent_training_group": "diverse_public",
            "formal_eval": False,
            "training_eligible": True,
        }
        for index in range(128)
    ]
    receipt = train_pure_rl._assert_r274_r195_research_jobs(
        rows,
        expected_content_digest=digest,
        iteration=3,
    )
    assert receipt["games"] == 128
    assert receipt["learner_first_games"] == 64
    assert receipt["learner_second_games"] == 64

    rows[0]["opponent_content_digest"] = "sha256:" + "e" * 64
    with pytest.raises(RuntimeError, match="identity changed"):
        train_pure_rl._assert_r274_r195_research_jobs(
            rows,
            expected_content_digest=digest,
            iteration=3,
        )


def test_r260_scheduled_rehearsal_receipt_requires_bounded_gradient_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "canary.pt"
    post_checkpoint = tmp_path / "rehearsed.pt"
    index = tmp_path / "sidecar.sqlite3"
    inzi_dataset = tmp_path / "joined.jsonl.gz"
    checkpoint.write_bytes(b"checkpoint")
    post_checkpoint.write_bytes(b"checkpoint child")
    inzi_dataset.write_bytes(b"joined sidecar")
    daily = {
        (date(2026, 7, 22) + timedelta(days=offset)).isoformat(): (
            "sha256:" + f"{offset:064x}"
        )
        for offset in range(20)
    }
    source_manifest = "sha256:" + "a" * 64
    db = sqlite3.connect(index)
    db.execute("CREATE TABLE metadata (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)")
    db.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("schema", json.dumps("poke_bot.r260_inzi_sidecar_index/v1")),
            ("source_manifest_sha256", json.dumps(source_manifest)),
            ("daily_meta_sha256s", json.dumps(daily, sort_keys=True, separators=(",", ":"))),
        ],
    )
    db.commit()
    db.close()
    index.chmod(0o444)
    record = {
        "schema": "poke_bot.r260_scheduled_expert_rehearsal/v1",
        "status": "passed",
        "owner_contract_sha256": "sha256:" + "1" * 64,
        "migration_receipt_sha256": "sha256:" + "2" * 64,
        "canary_activation_receipt_sha256": "sha256:" + "3" * 64,
        "sidecar_binding_sha256": "sha256:" + "4" * 64,
        "pre_checkpoint": train_pure_rl._r260_file_identity(checkpoint),
        "post_checkpoint": train_pure_rl._r260_file_identity(post_checkpoint),
        "index": train_pure_rl._r260_file_identity(index),
        "inzi_dataset": train_pure_rl._r260_file_identity(inzi_dataset),
        "expert_tactical_overlay": train_pure_rl._r260_file_identity(inzi_dataset),
        "index_provenance": {
            "schema": "poke_bot.r260_inzi_sidecar_index/v1",
            "source_manifest_sha256": source_manifest,
            "daily_meta_sha256s": daily,
        },
        "loss_weights": {
            "visible_tutor_completion": 0.025,
            "terminal_conversion": 0.025,
            "tactical_sequence_outcome": 0.025,
        },
        "rl_iteration_before_after": [4, 4],
        "bounded_batch_games": 8,
        "full_window_device_resident": False,
        "sampled_keys": [["ep", 0, 1, "sha256:" + "b" * 64]],
        "gradient_reachability": {name: True for name in ("own_deck_ledger_adapter.", "visible_tutor_completion_head.", "terminal_conversion_head.", "tactical_sequence_outcome_head.", "visible_tutor_completion_route.", "terminal_conversion_route.", "decision_fusion.")},
        "tactical_exact_root_count": 1024,
    }
    record["receipt_sha256"] = train_pure_rl._canonical_digest(record)
    assert (
        train_pure_rl._validate_r260_scheduled_rehearsal_receipt(record)[
            "post_checkpoint"
        ]["sha256"]
        == record["post_checkpoint"]["sha256"]
    )
    expected = {
        name: record[name]
        for name in (
            "owner_contract_sha256",
            "migration_receipt_sha256",
            "canary_activation_receipt_sha256",
            "sidecar_binding_sha256",
        )
    }
    expected["inzi_dataset_identity"] = dict(record["inzi_dataset"])
    expected["expert_tactical_overlay_identity"] = dict(
        record["expert_tactical_overlay"]
    )
    expected["source_manifest_sha256"] = source_manifest
    expected["daily_meta_sha256s"] = daily
    assert train_pure_rl._validate_r260_scheduled_rehearsal_receipt(
        record,
        expected_r260_inputs=expected,
    )["inzi_dataset"] == record["inzi_dataset"]

    missing_gradient = json.loads(json.dumps(record))
    missing_gradient["gradient_reachability"]["decision_fusion."] = False
    missing_gradient["receipt_sha256"] = train_pure_rl._canonical_digest(
        {
            key: value
            for key, value in missing_gradient.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="gradient"):
        train_pure_rl._validate_r260_scheduled_rehearsal_receipt(missing_gradient)

    post_checkpoint.write_bytes(b"tampered child")
    with pytest.raises(RuntimeError, match="FileIdentity"):
        train_pure_rl._validate_r260_scheduled_rehearsal_receipt(record)
    with pytest.raises(SystemExit):
        _r241_args("--expert-rehearsal-force-before", "5")
    with pytest.raises(SystemExit):
        _r241_args("--no-terminal-expert-rehearsal")


def test_r241_fixed_cycle_requires_paired_checkpoint_derived_evidence() -> None:
    with pytest.raises(SystemExit):
        train_pure_rl._parse_args(
            [
                "--iterations",
                "10",
                "--fixed-cycle-updates",
                "10",
                "--expert-rehearsal-every",
                "5",
                "--expert-rehearsal-epochs",
                "5",
                "--terminal-expert-rehearsal",
            ]
        )
    with pytest.raises(SystemExit):
        train_pure_rl._parse_args(
            [
                "--r241-peak-r195-preservation-receipt",
                "/tmp/r241-peak-r195-preservation-v6.json",
                "--r241-peak-r195-preservation-receipt-sha256",
                "sha256:" + "a" * 64,
            ]
        )


def test_r241_fixed_cycle_gate_passes_do_not_request_a_stop() -> None:
    assert train_pure_rl._gate_pass_should_stop(
        fixed_cycle_updates=10,
        continue_after_gate=False,
        terminal_target_reached=True,
    ) is False
    assert train_pure_rl._gate_pass_should_stop(
        fixed_cycle_updates=0,
        continue_after_gate=False,
        terminal_target_reached=True,
    ) is True


def test_r241_boundary_five_refresh_is_dispatched_before_next_collection() -> None:
    source = inspect.getsource(train_pure_rl.run_full_loop)
    boundary = source.rindex("if next_it < int(args.iterations):")
    refresh = source.index(
        "_prepare_r241_precollection_refresh(",
        boundary,
    )
    collect = source.index("pending_collect = _kick_collect(", boundary)

    assert boundary < refresh < collect


def test_r260_branch_streams_before_the_unchanged_resident_rehearsal_path() -> None:
    source = inspect.getsource(train_pure_rl.run_full_loop)
    r260_branch = source.index(
        "if r260_own_deck_training_inputs:",
        source.index("def _prepare_expert_rehearsal"),
    )
    resident_path = source.index("# A completed rehearsal is immutable training evidence.")
    r260_source = source[r260_branch:resident_path]

    assert "streaming_r260_host_rehearsal_step(" in r260_source
    assert "manifest_workers=int(args.expert_manifest_workers)" in r260_source
    assert "R260InziSidecarIndex.build(" in r260_source
    assert "expert_cache.prepare(" not in r260_source
    assert resident_path < source.index("expert_cache.prepare(", resident_path)


def test_r280_fixed_cycle_refresh_reuses_the_gpu_resident_pack() -> None:
    source = inspect.getsource(train_pure_rl.run_full_loop)
    r280_branch = source.index("if args.r280_contiguous_expert_pack is not None:")
    r260_branch = source.index("if r260_own_deck_training_inputs:", r280_branch)
    resident_path = source.index("# A completed rehearsal is immutable training evidence.")
    r280_source = source[r280_branch:r260_branch]

    assert "run_r280_gpu_resident_refresh(" in r280_source
    assert "streaming_r260_host_rehearsal_step(" not in r280_source
    assert r280_branch < r260_branch < resident_path


def test_r241_fixed_cycle_requires_commits_00000_through_00009_only(
    tmp_path: Path,
) -> None:
    state = _completed_r241_state(tmp_path)

    assert train_pure_rl._assert_fixed_cycle_completion(
        tmp_path,
        state,
        updates=10,
    ) == {
        "updates_completed": 10,
        "last_completed_iteration": 9,
        "next_collection_started": False,
    }

    plan_dir = tmp_path / "collection_plans"
    plan_dir.mkdir()
    (plan_dir / "iter_00010.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="iter_00010"):
        train_pure_rl._assert_fixed_cycle_completion(tmp_path, state, updates=10)

    (plan_dir / "iter_00010.json").unlink()
    (tmp_path / "commits" / "iter_00010.json").write_text(
        json.dumps({"last_completed_iteration": 10, "next_iteration": 11}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly immutable commits"):
        train_pure_rl._assert_fixed_cycle_completion(tmp_path, state, updates=10)
