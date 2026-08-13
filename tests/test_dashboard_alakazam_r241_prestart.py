"""Receipt-backed dashboard projection for the r241 successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import dashboard_snapshot


def _write_receipted_day(staging: Path, day: str) -> None:
    day_root = staging / "daily" / day
    receipt_root = staging / "receipts"
    day_root.mkdir(parents=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    meta = day_root / "meta.json"
    shard = day_root / "own_deck_rollouts.jsonl.gz"
    meta.write_bytes(b"{}\n")
    shard.write_bytes(b"test-shard")
    meta.chmod(0o444)
    shard.chmod(0o444)
    day_root.chmod(0o555)
    payload = {
        "schema": "poke_bot.r260_own_deck_prefix_transport_receipt/v1",
        "status": "staged_non_eligible",
        "day": day,
        "staging_training_eligible": False,
        "joined_dataset_created": False,
        "final_binding_created": False,
        "canonical_root_exists": False,
        "destination_files": {
            "meta": {"size_bytes": meta.stat().st_size},
            "shard": {"size_bytes": shard.stat().st_size},
        },
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    payload["receipt_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    (receipt_root / f"{day}.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n"
    )


def test_r274_formal_evaluation_projects_latest_completed_game() -> None:
    log = (
        "pure_rl heldout:strong_public_gate iter=0:  16%|x| 731/4500 "
        "[13:30<1:04:14, 1.02s/game, rsock=0, wr=78.80%/731g (gate 75%)]\r"
        "pure_rl heldout:strong_public_gate iter=0:  16%|x| 732/4500 "
        "[13:31<1:04:13, 1.02s/game, rsock=0, wr=78.85%/733g (gate 75%)]\r"
    )

    progress = dashboard_snapshot._latest_r274_formal_evaluation_progress(
        log, iteration=0
    )

    assert progress == {
        "current": 733,
        "total": 4500,
        "percent": 100.0 * 733 / 4500,
        "win_rate_percent": 78.85,
        "win_rate_games": 733,
        "gate_percent": 75.0,
    }


def test_r274_formal_evaluation_rejects_wrong_iteration_or_invalid_count() -> None:
    wrong_iteration = (
        "pure_rl heldout:strong_public_gate iter=1: 1%|x| 4/4500 "
        "[00:01<?, wr=75.00%/4g (gate 75%)]"
    )
    invalid = (
        "pure_rl heldout:strong_public_gate iter=0: 1%|x| 4/4500 "
        "[00:01<?, wr=75.00%/4501g (gate 75%)]"
    )

    assert (
        dashboard_snapshot._latest_r274_formal_evaluation_progress(
            wrong_iteration, iteration=0
        )
        is None
    )
    assert (
        dashboard_snapshot._latest_r274_formal_evaluation_progress(
            invalid, iteration=0
        )
        is None
    )


def test_r274_formal_evaluation_accepts_one_game_async_counter_skew() -> None:
    log = (
        "pure_rl heldout:strong_public_gate iter=0: 28%|x| 1239/4500 "
        "[23:20<1:02:18, 1.15s/game, wr=78.50%/1237g (gate 75%)]"
    )

    progress = dashboard_snapshot._latest_r274_formal_evaluation_progress(
        log, iteration=0
    )

    assert progress is not None
    assert progress["current"] == 1239
    assert progress["win_rate_games"] == 1237
    assert progress["win_rate_percent"] == 78.5


def test_r241_projection_uses_contiguous_immutable_receipt_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_new_list_direct_policy_r274/v1",
                "candidate_id": "alakazam-new-list-direct-policy-r274",
                "latest_owner_clarification_revision": 285,
                "twenty_update_horizon_override": {
                    "owner_revision": 285,
                    "rl_updates_exact": 20,
                    "next_iteration_after_loop": 20,
                    "iteration_20_collection_allowed": False,
                    "expert_refresh_and_submission_boundaries": [5, 10, 15, 20],
                    "total_submission_count_including_bootstrap": 5,
                },
            }
        )
    )
    staging = tmp_path / "staging"
    _write_receipted_day(staging, "2026-07-22")
    _write_receipted_day(staging, "2026-07-23")
    # A later day cannot skip the missing July 24 receipt.
    _write_receipted_day(staging, "2026-07-25")
    monkeypatch.setattr(
        dashboard_snapshot,
        "unit_state",
        lambda *_args, **_kwargs: {
            "active": False,
            "pid": 0,
            "started": False,
            "active_state": "inactive",
            "sub_state": "dead",
        },
    )

    result = dashboard_snapshot.alakazam_r241_prestart_progress(
        contract_path=contract,
        staging_root=staging,
        final_root=tmp_path / "final",
        source_materialization={"active": True, "committed_day_count": 19},
    )

    assert result["selected"] is True
    assert result["status"] == "materializing"
    assert result["current"] == 2
    assert result["total"] == 20
    assert [row["day"] for row in result["transfer"]["days"]] == [
        "2026-07-22",
        "2026-07-23",
    ]
    assert result["schedule"]["rl_updates"] == 20
    assert result["schedule"]["soft_refresh_boundaries"] == [5, 10, 15, 20]
    assert result["schedule"]["submission_count"] == 5
    assert result["schedule"]["r195_submission_55378392_minimum"] == 128
    assert result["model_plan"]["heads_training_active"] == 21
    assert result["model_plan"]["bootstrap_fusion_routes_active"] == 21
    assert result["model_plan"]["post_bootstrap_fusion_routes_active"] == 21
    assert result["model_plan"]["tactical_sequence_head_present"] is False
    assert (
        result["model_plan"]["tactical_sequence_route"]
        == "removed_before_rl_update_0"
    )
    assert result["model_plan"]["tactical_route_active"] is False
    assert result["mode"] == "alakazam_new_list_direct_r274_prestart"
    assert result["run"] == "alakazam_new_list_direct_policy_r274"


def test_r241_projection_rejects_tampered_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_new_list_direct_policy_r274/v1",
                "candidate_id": "alakazam-new-list-direct-policy-r274",
                "latest_owner_clarification_revision": 275,
            }
        )
    )
    staging = tmp_path / "staging"
    _write_receipted_day(staging, "2026-07-22")
    receipt = staging / "receipts" / "2026-07-22.json"
    payload = json.loads(receipt.read_text())
    payload["joined_dataset_created"] = True
    receipt.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(
        dashboard_snapshot,
        "unit_state",
        lambda *_args, **_kwargs: {"active": False, "pid": 0},
    )

    result = dashboard_snapshot.alakazam_r241_prestart_progress(
        contract_path=contract,
        staging_root=staging,
        final_root=tmp_path / "final",
    )

    assert result["current"] == 0
    assert result["transfer"]["days"] == []


def test_r274_active_bootstrap_reports_exact_read_only_stream_cursor(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    first = feature_root / "day-01.features"
    second = feature_root / "day-02.features"
    first.write_bytes(b"a" * 100)
    second.write_bytes(b"b" * 300)
    manifest = feature_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {"path": first.name, "bytes": 100},
                    {"path": second.name, "bytes": 300},
                ]
            }
        )
    )
    proc_root = tmp_path / "proc"
    fd_root = proc_root / "42" / "fd"
    fdinfo_root = proc_root / "42" / "fdinfo"
    fd_root.mkdir(parents=True)
    fdinfo_root.mkdir(parents=True)
    (fd_root / "7").symlink_to(second)
    (fdinfo_root / "7").write_text("pos:\t150\n")
    trainer = {
        "active": True,
        "pid": 42,
        "exec_start": (
            "python scripts/run_r274_streaming_bootstrap.py "
            f"--manifest {manifest} --epochs 25"
        ),
    }

    progress = dashboard_snapshot.r274_streaming_bootstrap_progress(
        trainer, proc_root=proc_root
    )

    assert progress["available"] is True
    assert progress["epochs_target"] == 25
    assert progress["epoch"] is None
    assert progress["epoch_status"] == "not_emitted_by_current_process"
    assert progress["active_shard"] == 2
    assert progress["shard_count"] == 2
    assert progress["active_shard_percent"] == 50.0
    assert progress["current"] == 250
    assert progress["total"] == 400
    assert progress["percent"] == 62.5


def test_r280_gpu_resident_bootstrap_reports_optimizer_heartbeat(
    monkeypatch,
) -> None:
    trainer = {
        "active": True,
        "pid": 4026913,
        "exec_start": "python scripts/run_r280_gpu_resident_bootstrap.py --epochs 25",
    }
    monkeypatch.setattr(
        dashboard_snapshot.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                b"[r280-gpu-pack] resident tensor_gib=5.332\r"
                b"expert rehearsal before iter0 ep1/25:  74%|x| "
                b"750/1008 [03:06<01:02, step=71978, loss=1.249]\r"
                b"expert rehearsal before iter0 ep1/25:  75%|x| "
                b"752/1008 [03:06<01:03, step=71979, loss=1.280]\r"
            )
        ),
    )

    progress = dashboard_snapshot.r274_bootstrap_progress(trainer)

    assert progress["available"] is True
    assert progress["phase"] == "gpu_resident_bootstrap"
    assert progress["epoch"] == 1
    assert progress["epochs_target"] == 25
    assert progress["batch"] == 752
    assert progress["batches_per_epoch"] == 1008
    assert progress["optimizer_step"] == 71979
    assert progress["current"] == 752 / 1008
    assert progress["total"] == 25
    assert progress["percent"] == 100.0 * (752 / 1008) / 25
    assert "full corpus resident on cuda:1" in progress["latest_line"]


def test_r281_matchup_adapter_bootstrap_reports_real_optimizer_epochs(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = {
        "active": True,
        "pid": 43098,
        "exec_start": "python scripts/train_r281_bootstrap_matchup_adapters.py",
    }
    monkeypatch.setattr(
        dashboard_snapshot,
        "ALAKAZAM_R281_ADAPTER_BOOTSTRAP_RECEIPT",
        tmp_path / "absent.json",
    )
    monkeypatch.setattr(
        dashboard_snapshot.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                b"[r281-adapters] epoch=2/25 steps=831 "
                b"rows=1848578 loss=0.881865\n"
                b"[r281-adapters] epoch=3/25 steps=829 "
                b"rows=1848578 loss=0.881004\n"
            )
        ),
    )

    progress = dashboard_snapshot.r281_matchup_adapter_bootstrap_progress(
        trainer
    )

    assert progress["available"] is True
    assert progress["active"] is True
    assert progress["phase"] == "matchup_adapter_bootstrap"
    assert progress["epoch"] == 3
    assert progress["epochs_target"] == 25
    assert progress["optimizer_steps_latest_epoch"] == 829
    assert progress["routed_decisions_latest_epoch"] == 1_848_578
    assert progress["percent"] == 12.0
    assert "optimizer active" in progress["latest_line"]
