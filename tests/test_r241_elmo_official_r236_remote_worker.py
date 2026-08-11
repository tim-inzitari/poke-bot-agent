"""Focused safety tests for the isolated r241 Elmo :8767 worker contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot import remote_jobs
from poke_bot import r241_elmo_official_r236_remote_worker as worker
from scripts import launch_r241_elmo_official_r236_worker, run_remote_worker

ROOT = Path(__file__).resolve().parents[1]


def _file(path: Path, payload: str = "payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _readonly_json(path: Path, payload: object) -> Path:
    _file(path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o444)
    return path


def _baseline_payload(tmp_path: Path) -> worker.R241ElmoBaselinePayload:
    root = tmp_path / "alakazam-new-list-direct-r241-baselines-test"
    root.mkdir(exist_ok=True)
    manifest = _file(root / "r241-baseline-payload-manifest.json", "{}")
    staging = _file(tmp_path / "runtime" / "baseline-payload-staging.json", "{}")
    canonical = _file(tmp_path / "runtime" / "canonical-baseline-roster.json", "{}")
    return worker.R241ElmoBaselinePayload(
        root=root,
        manifest=manifest,
        manifest_sha256="sha256:" + "4" * 64,
        baseline_tree_sha256="sha256:" + "5" * 64,
        file_inventory_sha256="sha256:" + "6" * 64,
        staging_receipt=staging,
        staging_receipt_sha256="sha256:" + "7" * 64,
        canonical_roster_receipt=canonical,
        canonical_roster_receipt_sha256="sha256:" + "8" * 64,
        baseline_manifest_sha256="sha256:" + "9" * 64,
        baseline_roster_sha256="sha256:" + "a" * 64,
    )


def _checkpoint_transport(tmp_path: Path) -> worker.R241ElmoCheckpointTransport:
    staging = _file(tmp_path / "runtime" / "checkpoint-transport-staging.json", "{}")
    digest = "sha256:" + "b" * 64
    return worker.R241ElmoCheckpointTransport(
        host_root=tmp_path / "elmo-checkpoint-transport",
        container_root=worker.ELMO_R241_CHECKPOINT_TRANSPORT_CONTAINER_ROOT,
        staging_receipt=staging,
        staging_receipt_sha256="sha256:" + "c" * 64,
        initial_checkpoint=Path("/workspace/checkpoint/model." + "b" * 16 + ".pt"),
        initial_checkpoint_sha256=digest,
    )


def _receipt_preflight(tmp_path: Path) -> worker.R241ElmoPreflight:
    snapshot_root = tmp_path / "alakazam-new-list-direct-r241-src-test"
    snapshot_root.mkdir()
    snapshot_manifest = _file(snapshot_root / "r241-source-snapshot-manifest.json", "{}")
    cg_root = tmp_path / "cg-r236"
    cg_root.mkdir()
    _file(cg_root / "r241_official_libcg_direct_policy_preflight.json", "{}")
    checkpoint_transport = _checkpoint_transport(tmp_path)
    checkpoint = checkpoint_transport.initial_checkpoint
    marker = _file(tmp_path / "checkpoints" / "matchup-runtime-activation.json", "{}")
    learner_tree = Path("/workspace/checkpoint/r195-e60.json")
    adapter = _file(tmp_path / "runtime" / "marnie-adapter.json", "{}")
    baseline_payload = _baseline_payload(tmp_path)
    h10_tree = _file(baseline_payload.root / "specialists" / "marnie-tree.json")
    return worker.R241ElmoPreflight(
        repo_root=snapshot_root,
        source_snapshot_manifest=snapshot_manifest,
        checkpoint=checkpoint,
        cg_lib_path=cg_root,
        adapter_receipt=adapter,
        learner_matchup_tree=learner_tree,
        matchup_runtime_marker=marker,
        h10_matchup_tree=h10_tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        environment={worker.ELMO_R241_WORKER_IMAGE_ID_ENV: "sha256:" + "e" * 64},
        sources={
            "source_snapshot": {
                "schema": worker.R241_SOURCE_SNAPSHOT_SCHEMA,
                "status": "authenticated_immutable_source_snapshot",
                "authenticated": True,
                "root": str(snapshot_root),
                "source_execution_root": str(snapshot_root),
                "manifest": str(snapshot_manifest),
                "manifest_sha256": worker._sha256(snapshot_manifest),
                "source_tree_sha256": "sha256:" + "2" * 64,
                "file_inventory_sha256": "sha256:" + "3" * 64,
                "outputs_root": str(worker.ELMO_OUTPUTS_ROOT),
            },
            "owner_contract": {"path": "state/r241.json", "sha256": "sha256:" + "1" * 64},
            "matchup_archetype_refresh": {
                "status": "deferred_not_required",
                "required_for_r241_activation": False,
                "slot_migration_status": "no_slot_change",
                "baseline_slot_registry_sha256": worker.BASELINE_ADAPTER_ROSTER_SHA256,
                "immutable_slot_prefix": 20,
                "new_slots": [],
            },
        },
        matchup_runtime={
            "adapter_format": "poke-bot-matchup-adapter-bank-v6",
            "route_target_ids": ["alakazam"],
            "route_physical_slots": [0],
        },
    )


def _source_staging_receipt_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
    verified_from_published_immutable_root: bool | None,
) -> tuple[dict[str, object], Path, str, str]:
    """Build the create-only source-staging shape consumed by Elmo preflight."""

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(worker, "ELMO_OUTPUTS_ROOT", outputs)
    source_root = tmp_path / "alakazam-new-list-direct-r241-src-final"
    source_root.mkdir()
    manifest = _file(source_root / "r241-source-snapshot-manifest.json", "{}")
    owner_contract_sha256 = "sha256:" + "1" * 64
    source_snapshot: dict[str, object] = {
        "schema": worker.R241_SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": str(source_root),
        "source_execution_root": str(source_root),
        "manifest": str(manifest),
        "manifest_sha256": worker._sha256(manifest),
        "source_tree_sha256": "sha256:" + "2" * 64,
        "file_inventory_sha256": "sha256:" + "3" * 64,
        "owner_contract_sha256": owner_contract_sha256,
        "outputs_root": str(outputs),
    }
    closure: dict[str, object] = {
        "baseline_payloads_separate_and_receipted": True,
    }
    if verified_from_published_immutable_root is not None:
        closure["verified_from_published_immutable_root"] = (
            verified_from_published_immutable_root
        )
    receipt = _readonly_json(
        outputs / "state" / "r241-source-staging.json",
        {
            "schema": worker.R241_SOURCE_SNAPSHOT_STAGING_SCHEMA,
            "revision": worker.R241_REVISION,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "passed",
            "passed": True,
            "operation": operation,
            "source_snapshot": source_snapshot,
            "closure": closure,
        },
    )
    return source_snapshot, receipt, worker._sha256(receipt), owner_contract_sha256


@pytest.mark.parametrize(
    ("operation", "verified_from_published_immutable_root"),
    (
        ("deterministic_stage_or_verify", None),
        # This is the final published-root receipt shape produced by
        # stage_r241_source_snapshot.py after immutable host verification.
        ("verify_published_immutable_source_snapshot", True),
    ),
)
def test_r241_elmo_accepts_only_trusted_source_staging_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    verified_from_published_immutable_root: bool | None,
) -> None:
    source_snapshot, receipt, receipt_sha256, owner_contract_sha256 = (
        _source_staging_receipt_shape(
            tmp_path,
            monkeypatch,
            operation=operation,
            verified_from_published_immutable_root=verified_from_published_immutable_root,
        )
    )

    validated = worker._source_snapshot_from_staging_receipt(
        source_snapshot=source_snapshot,
        owner_contract_sha256=owner_contract_sha256,
        staging_receipt=receipt,
        staging_receipt_sha256=receipt_sha256,
    )

    assert validated["staging_receipt"] == str(receipt)
    assert validated["staging_receipt_sha256"] == receipt_sha256


@pytest.mark.parametrize(
    ("operation", "verified_from_published_immutable_root"),
    (
        ("unrecognized_source_stage_operation", True),
        ("verify_published_immutable_source_snapshot", None),
        ("verify_published_immutable_source_snapshot", False),
    ),
)
def test_r241_elmo_rejects_untrusted_or_unverified_source_staging_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    verified_from_published_immutable_root: bool | None,
) -> None:
    source_snapshot, receipt, receipt_sha256, owner_contract_sha256 = (
        _source_staging_receipt_shape(
            tmp_path,
            monkeypatch,
            operation=operation,
            verified_from_published_immutable_root=verified_from_published_immutable_root,
        )
    )

    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="source staging receipt binding drifted",
    ):
        worker._source_snapshot_from_staging_receipt(
            source_snapshot=source_snapshot,
            owner_contract_sha256=owner_contract_sha256,
            staging_receipt=receipt,
            staging_receipt_sha256=receipt_sha256,
        )


def _baseline_staging_receipt_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    canonical_path_field: str = "path",
) -> tuple[Path, str, Path, str, str]:
    """Build the immutable baseline staging shape emitted by its stager."""

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(worker, "ELMO_OUTPUTS_ROOT", outputs)
    baseline_root = tmp_path / "alakazam-new-list-direct-r241-baselines-final"
    baseline_root.mkdir()
    manifest = _file(baseline_root / "r241-baseline-payload-manifest.json", "{}")
    canonical_roster = _readonly_json(
        outputs / "state" / "r241_canonical_baseline_roster_receipt.v1.json",
        {"fixture": "canonical-roster"},
    )
    canonical_roster_sha256 = worker._sha256(canonical_roster)
    owner_contract_sha256 = "sha256:" + "1" * 64
    baseline_manifest_sha256 = "sha256:" + "5" * 64
    baseline_roster_sha256 = "sha256:" + "6" * 64
    snapshot = {
        "schema": worker.baseline_payload_snapshot.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": worker.R241_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "authenticated": True,
        "host": "elmo",
        "root": str(baseline_root),
        "manifest": str(manifest),
        "manifest_sha256": worker._sha256(manifest),
        "baseline_tree_sha256": "sha256:" + "2" * 64,
        "file_inventory_sha256": "sha256:" + "3" * 64,
        "owner_contract_sha256": owner_contract_sha256,
        "canonical_roster_receipt": str(canonical_roster),
        "canonical_roster_receipt_sha256": canonical_roster_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "baseline_roster_sha256": baseline_roster_sha256,
    }
    canonical = {
        canonical_path_field: str(canonical_roster),
        "sha256": canonical_roster_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "baseline_roster_sha256": baseline_roster_sha256,
    }
    staging = _readonly_json(
        outputs / "state" / "r241-baseline-payload-staging-elmo.v1.json",
        {
            "schema": worker.baseline_payload_snapshot.BASELINE_PAYLOAD_STAGING_SCHEMA,
            "revision": worker.R241_REVISION,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "passed",
            "passed": True,
            "operation": "deterministic_stage_or_verify",
            "receipt_outside_source_and_baseline_snapshot": True,
            "baseline_payload_snapshot": snapshot,
            # The canonical subobject's production shape is {path, sha256,
            # baseline_manifest_sha256, baseline_roster_sha256}.
            "canonical_roster_receipt": canonical,
        },
    )
    return (
        staging,
        worker._sha256(staging),
        canonical_roster,
        canonical_roster_sha256,
        owner_contract_sha256,
    )


def test_r241_elmo_accepts_actual_baseline_staging_canonical_path_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        staging,
        staging_sha256,
        canonical_roster,
        canonical_roster_sha256,
        owner_contract_sha256,
    ) = _baseline_staging_receipt_shape(tmp_path, monkeypatch)

    contract = worker._baseline_payload_contract_from_staging_receipt(
        owner_contract_sha256=owner_contract_sha256,
        staging_receipt=staging,
        staging_receipt_sha256=staging_sha256,
        canonical_roster_receipt=canonical_roster,
        canonical_roster_receipt_sha256=canonical_roster_sha256,
    )

    assert contract["canonical_roster_receipt"] == str(canonical_roster)
    assert contract["canonical_roster_receipt_sha256"] == canonical_roster_sha256


def test_r241_elmo_rejects_legacy_baseline_staging_canonical_field_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        staging,
        staging_sha256,
        canonical_roster,
        canonical_roster_sha256,
        owner_contract_sha256,
    ) = _baseline_staging_receipt_shape(
        tmp_path,
        monkeypatch,
        canonical_path_field="canonical_roster_receipt",
    )

    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="baseline staging receipt omits path",
    ):
        worker._baseline_payload_contract_from_staging_receipt(
            owner_contract_sha256=owner_contract_sha256,
            staging_receipt=staging,
            staging_receipt_sha256=staging_sha256,
            canonical_roster_receipt=canonical_roster,
            canonical_roster_receipt_sha256=canonical_roster_sha256,
        )


def test_r241_elmo_endpoint_is_literal_and_rejects_legacy_defaults() -> None:
    assert worker.assert_r241_elmo_endpoint("192.168.1.143:8767") == "192.168.1.143:8767"
    for endpoint in (
        "192.168.1.143:8765",
        "192.168.1.158:8766",
        "bert.local:8766",
        "",
        "elmo:8767",
    ):
        with pytest.raises(worker.R241ElmoRemoteWorkerError):
            worker.assert_r241_elmo_endpoint(endpoint)


def test_r241_environment_forces_remote_diverse_mix_and_rejects_planners(
    tmp_path: Path,
) -> None:
    cg_root = tmp_path / "cg"
    cg_root.mkdir()
    adapter = _file(tmp_path / "adapter.json")
    tree = _file(tmp_path / "tree.json")
    baseline_payload = _baseline_payload(tmp_path)
    checkpoint_transport = _checkpoint_transport(tmp_path)
    environment = worker.build_r241_elmo_collection_environment(
        cg_lib_path=cg_root,
        adapter_receipt=adapter,
        learner_matchup_tree=tree,
        baseline_payload=baseline_payload,
        checkpoint_transport=checkpoint_transport,
        adapter_format="poke-bot-matchup-adapter-bank-v6",
        environment={
            "PATH": "/usr/bin",
            worker.ELMO_R241_WORKER_IMAGE_ID_ENV: "sha256:" + "e" * 64,
            # Match the sealed Compose environment: this exact zero disables
            # the recursive planner and is the only admitted token exception.
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        },
    )
    assert environment["PURE_RL_PUBLIC_MIX_LOCAL_ONLY"] == "0"
    assert environment["POKEBOT_REMOTE_ALLOWED_JOB_KINDS"] == (
        "play,self_play,self_play_multi,runtime_probe"
    )
    assert environment["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] == "1"
    assert environment["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "0"
    assert environment["POKEBOT_REMOTE_WORKER_ARM_FILE"] == str(
        worker.ELMO_REMOTE_ARM_FILE
    )
    assert environment["POKEBOT_BASELINES_DIR"] == str(baseline_payload.root)
    assert environment["POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST"] == str(
        baseline_payload.manifest
    )
    assert environment[worker.ELMO_R241_CHECKPOINT_TRANSPORT_ENV] == "/workspace/checkpoint"
    assert environment[worker.ELMO_R241_WORKER_IMAGE_ID_ENV] == "sha256:" + "e" * 64

    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="worker image ID environment",
    ):
        worker.build_r241_elmo_collection_environment(
            cg_lib_path=cg_root,
            adapter_receipt=adapter,
            learner_matchup_tree=tree,
            baseline_payload=baseline_payload,
            checkpoint_transport=checkpoint_transport,
            adapter_format="poke-bot-matchup-adapter-bank-v6",
            environment={"PATH": "/usr/bin"},
        )

    for leaked in (
        {"POKEBOT_MCTS_SIMS": "64"},
        {"POKEBOT_RTP_CHECKPOINT": "/old.pt"},
        {"POKEBOT_LIBCG_PATH": "/old/cg"},
        {"POKEBOT_BATCH_LIBCG": ""},
        {"POKEBOT_SEARCH_TARGETS": "1"},
        {"MCTS_SIMS": "64"},
        {"RTP_PATH": "/old.pt"},
        {"POKEBOT_USE_RECURSIVE_TURN_PLANNER": "1"},
        {"POKEBOT_USE_RECURSIVE_TURN_PLANNER_SHADOW": "0"},
        {"LD_PRELOAD": "/old/libcg.so"},
        {"LIBCG_PATH": "/old/cg"},
        {"PURE_RL_PUBLIC_MIX_LOCAL_ONLY": "1"},
    ):
        with pytest.raises(worker.R241ElmoRemoteWorkerError):
            worker.build_r241_elmo_collection_environment(
                cg_lib_path=cg_root,
                adapter_receipt=adapter,
                learner_matchup_tree=tree,
                baseline_payload=baseline_payload,
                checkpoint_transport=checkpoint_transport,
                adapter_format="poke-bot-matchup-adapter-bank-v6",
                environment={
                    worker.ELMO_R241_WORKER_IMAGE_ID_ENV: "sha256:" + "e" * 64,
                    **leaked,
                },
            )


def test_r241_receipts_bind_host_runtime_gameplay_and_are_write_once(
    tmp_path: Path,
) -> None:
    preflight = _receipt_preflight(tmp_path)
    receipt_dir = tmp_path / "receipts"
    receipts = worker.build_r241_elmo_preflight_receipts(
        preflight, receipt_dir=receipt_dir
    )
    paths = worker.write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=receipt_dir
    )
    # An exact retry is allowed; evidence cannot be silently overwritten.
    assert worker.write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=receipt_dir
    ) == paths
    validated = worker.validate_r241_elmo_preflight_manifest(paths["manifest"])
    assert validated["payload"]["endpoint"]["literal"] == "192.168.1.143:8767"
    assert validated["runtime_receipt"]["environment"][
        "PURE_RL_PUBLIC_MIX_LOCAL_ONLY"
    ] == "0"
    assert validated["gameplay_receipt"]["collection_contract"][
        "public_mix_games_exact"
    ] == 7172
    assert validated["gameplay_receipt"]["promotion_jobs_allowed"] is False
    assert validated["runtime_receipt"]["environment"]["POKEBOT_BASELINES_DIR"] == str(
        preflight.baseline_payload.root
    )
    assert validated["runtime_receipt"]["direct_policy"][
        "runtime_call_counters_available"
    ] is False
    assert validated["gameplay_receipt"]["adapter_behavior"][
        "non_h10_public_opponent_selectors"
    ] == "preserved_external_public_opponents"

    altered = dict(receipts)
    altered["gameplay"] = dict(receipts["gameplay"])
    altered["gameplay"]["promotion_jobs_allowed"] = True
    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="differs"):
        worker.write_r241_elmo_preflight_receipts(altered, receipt_dir=receipt_dir)


def test_r241_manifest_rejects_a_tampered_public_mix_receipt(tmp_path: Path) -> None:
    preflight = _receipt_preflight(tmp_path)
    receipt_dir = tmp_path / "receipts"
    receipts = worker.build_r241_elmo_preflight_receipts(
        preflight, receipt_dir=receipt_dir
    )
    paths = worker.write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=receipt_dir
    )
    gameplay = json.loads(paths["gameplay"].read_text(encoding="utf-8"))
    gameplay["collection_contract"]["public_mix_local_only"] = True
    paths["gameplay"].write_text(
        json.dumps(gameplay, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="receipt drifted"):
        worker.validate_r241_elmo_preflight_manifest(paths["manifest"])


def test_r241_preflight_refuses_workspace_path_remapping() -> None:
    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="container remap"):
        worker._require_elmo_absolute_path(
            Path("/workspace/r241-runtime"), label="test path"
        )


def test_r241_cli_refuses_noncanonical_receipt_output_path(tmp_path: Path) -> None:
    args = type(
        "Args", (), {"receipt_dir": tmp_path / "not-the-elmo-runtime"}
    )()

    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="canonical Elmo path"):
        launch_r241_elmo_official_r236_worker._run_preflight(args)


def test_r241_cli_forwards_the_required_checkpoint_transport_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_dir = tmp_path / "canonical-elmo-receipts"
    captured: dict[str, object] = {}
    sentinel = SimpleNamespace(repo_root=launch_r241_elmo_official_r236_worker.ROOT)
    manifest = {"path": str(tmp_path / "preflight-manifest.json"), "sha256": "sha256:" + "d" * 64}

    def _preflight(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        launch_r241_elmo_official_r236_worker,
        "DEFAULT_ELMO_RECEIPT_DIR",
        receipt_dir,
    )
    monkeypatch.setattr(
        launch_r241_elmo_official_r236_worker,
        "preflight_r241_elmo_remote_collection",
        _preflight,
    )
    monkeypatch.setattr(
        launch_r241_elmo_official_r236_worker,
        "build_r241_elmo_preflight_receipts",
        lambda _preflight, *, receipt_dir: {"manifest": {}},
    )
    monkeypatch.setattr(
        launch_r241_elmo_official_r236_worker,
        "write_r241_elmo_preflight_receipts",
        lambda _receipts, *, receipt_dir: {"manifest": tmp_path / "preflight-manifest.json"},
    )
    monkeypatch.setattr(
        launch_r241_elmo_official_r236_worker,
        "validate_r241_elmo_preflight_manifest",
        lambda _path: manifest,
    )
    transport_root = tmp_path / "checkpoint-transport"
    transport_receipt = tmp_path / "checkpoint-transport-staging.json"
    args = SimpleNamespace(
        receipt_dir=receipt_dir,
        endpoint="192.168.1.143:8767",
        source_snapshot_root=tmp_path / "source",
        source_snapshot_manifest=tmp_path / "source" / "r241-source-snapshot-manifest.json",
        checkpoint=Path("/workspace/checkpoint/model.pt"),
        cg_lib_path=tmp_path / "cg-r236",
        adapter_receipt=tmp_path / "marnie-adapter.json",
        learner_matchup_tree=Path("/workspace/checkpoint/r195-e60.json"),
        baselines_root=tmp_path / "baselines",
        checkpoint_transport_host_root=transport_root,
        checkpoint_transport_staging_receipt=transport_receipt,
        checkpoint_transport_staging_receipt_sha256="sha256:" + "e" * 64,
        source_staging_receipt=tmp_path / "source-staging.json",
        source_staging_receipt_sha256="sha256:" + "f" * 64,
        baseline_staging_receipt=tmp_path / "baseline-staging.json",
        baseline_staging_receipt_sha256="sha256:" + "0" * 64,
        canonical_roster_receipt=tmp_path / "canonical-roster.json",
        canonical_roster_receipt_sha256="sha256:" + "1" * 64,
    )

    assert launch_r241_elmo_official_r236_worker._run_preflight(args) == (
        sentinel,
        manifest,
    )
    assert captured["checkpoint_transport_host_root"] == transport_root
    assert captured["checkpoint_transport_staging_receipt"] == transport_receipt
    assert captured["checkpoint_transport_staging_receipt_sha256"] == "sha256:" + "e" * 64


def test_r241_current_mutable_checkout_is_not_an_eligible_source_snapshot() -> None:
    with pytest.raises(worker.R241ElmoRemoteWorkerError):
        worker.validate_current_r241_sources(ROOT)


def test_r251_keeps_ptcgreplay_metadata_inert_and_requires_current_roster() -> None:
    policy = json.loads(
        (ROOT / "state/alakazam-new-list-direct-policy-r241.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (ROOT / "state/alakazam-new-list-direct-r241-runtime-registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert worker._sha256(
        ROOT / "state/alakazam-new-list-direct-policy-r241.json"
    ).startswith("sha256:")
    assert (
        policy["latest_owner_clarification_revision"]
        == worker.R241_LATEST_OWNER_CLARIFICATION_REVISION
    )
    exclusion = policy["search_and_planning_exclusion"]
    assert exclusion["mcts"] == "forbidden_for_scoped_direct_roles"
    assert exclusion["public_opponent_selector_change"] == "forbidden"
    assert exclusion["public_search_firewall"] == "not_introduced"
    assert (
        exclusion["scope"]["frozen_non_h10_diverse_public_opponent_packages_and_selectors"]
        == "preserve_unchanged_per_r245"
    )
    summary = worker._validate_deferred_matchup_archetype_refresh(
        policy=policy,
        registry=registry,
        source_root=ROOT,
    )
    assert summary["slot_migration_status"] == "no_slot_change"
    assert summary["active_slot_count"] == 20
    assert summary["new_slots"] == []
    assert registry["direct_policy"]["adapter_receipt_elmo"] == str(
        worker.ELMO_R241_RUN_ROOT
        / "runtime/marnie-h10-direct-policy-adapter-r251-v8.json"
    )
    assert registry["peak_r195_preservation"]["receipt_elmo"] == str(
        worker.ELMO_R241_RUN_ROOT / "runtime/peak-r195-preservation-v6.json"
    )

    # r248 defers PTCGReplay metadata from r241 activation, and r251 keeps
    # that preservation while narrowing no-search only to direct roles.  The
    # raw frozen roster, not a stale future projection, is the no-slot-change
    # evidence.
    registry["matchup_archetype_refresh"] = {
        "status": "future_metadata_only",
        "new_slots": [{"slot": 20}],
    }
    assert worker._validate_deferred_matchup_archetype_refresh(
        policy=policy,
        registry=registry,
        source_root=ROOT,
    )["slot_migration_status"] == "no_slot_change"


def test_r241_elmo_worker_rejects_h10_receipt_from_a_stale_source_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "alakazam-new-list-direct-r241-src-active"
    source_root.mkdir()
    source_manifest = _file(
        source_root / "r241-source-snapshot-manifest.json",
        '{"source":"active"}\n',
    )
    adapter_receipt = _readonly_json(
        tmp_path / "runtime/marnie-h10-direct-policy-adapter-r251-v8.json",
        {
            "offline_preflight": {
                "source_snapshot_root": str(source_root),
                "source_snapshot_manifest": str(source_manifest),
                "source_snapshot_manifest_sha256": "sha256:" + "f" * 64,
                "native_function_calls": 0,
                "search_calls_made": 0,
                "simulator_battles_started": 0,
                "model_weights_loaded": False,
                "baseline_package_main_imported": False,
            }
        },
    )

    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="does not bind the active source snapshot",
    ):
        worker._validate_h10_adapter(
            adapter_receipt,
            environment={},
            source_snapshot={
                "root": str(source_root),
                "manifest": str(source_manifest),
                "manifest_sha256": worker._sha256(source_manifest),
            },
        )


def test_r241_manifest_rejects_any_future_slot_activation_for_this_cycle(
    tmp_path: Path,
) -> None:
    preflight = _receipt_preflight(tmp_path)
    receipts = worker.build_r241_elmo_preflight_receipts(
        preflight, receipt_dir=tmp_path / "receipts"
    )
    runtime = dict(receipts["runtime"])
    adapter = dict(runtime["matchup_adapter"])
    adapter["new_slots"] = [{"slot": 20, "archetype_id": "future"}]
    runtime["matchup_adapter"] = adapter
    receipts["runtime"] = runtime
    manifest = dict(receipts["manifest"])
    manifest_receipts = dict(manifest["receipts"])
    runtime_ref = dict(manifest_receipts["runtime"])
    runtime_ref["sha256"] = worker._json_digest(runtime)
    manifest_receipts["runtime"] = runtime_ref
    manifest["receipts"] = manifest_receipts
    receipts["manifest"] = manifest
    paths = worker.write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=tmp_path / "receipts"
    )
    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="Matchup Adapter receipt"):
        worker.validate_r241_elmo_preflight_manifest(paths["manifest"])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_main_optimizer_included", True),
        ("matchup_adapter_bank_preserved", False),
        ("matchup_adapter_external_terminal_runtime_enabled", False),
    ),
)
def test_r241_manifest_rejects_adapter_isolation_or_preservation_drift(
    tmp_path: Path, field: str, value: bool
) -> None:
    preflight = _receipt_preflight(tmp_path)
    receipts = worker.build_r241_elmo_preflight_receipts(
        preflight, receipt_dir=tmp_path / "receipts"
    )
    runtime = dict(receipts["runtime"])
    adapter = dict(runtime["matchup_adapter"])
    adapter[field] = value
    runtime["matchup_adapter"] = adapter
    receipts["runtime"] = runtime
    manifest = dict(receipts["manifest"])
    manifest_receipts = dict(manifest["receipts"])
    runtime_ref = dict(manifest_receipts["runtime"])
    runtime_ref["sha256"] = worker._json_digest(runtime)
    manifest_receipts["runtime"] = runtime_ref
    manifest["receipts"] = manifest_receipts
    receipts["manifest"] = manifest
    paths = worker.write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=tmp_path / "receipts"
    )
    with pytest.raises(worker.R241ElmoRemoteWorkerError, match="Matchup Adapter receipt"):
        worker.validate_r241_elmo_preflight_manifest(paths["manifest"])


def test_generic_remote_worker_remains_legacy_compatible_but_can_be_restricted() -> None:
    assert run_remote_worker._configured_remote_job_kinds({}) == (
        "play",
        "promotion",
        "self_play",
        "self_play_multi",
        "runtime_probe",
    )
    assert run_remote_worker._configured_remote_job_kinds(
        {"POKEBOT_REMOTE_ALLOWED_JOB_KINDS": "runtime_probe,play,self_play"}
    ) == ("play", "self_play", "runtime_probe")
    capabilities = run_remote_worker._configured_remote_worker_capabilities(
        {"POKEBOT_REMOTE_WORKER_CAPABILITY_TAGS": "r241_direct_policy_collection_v1"}
    )
    assert "r241_direct_policy_collection_v1" in capabilities
    with pytest.raises(ValueError, match="unsupported"):
        run_remote_worker._configured_remote_job_kinds(
            {"POKEBOT_REMOTE_ALLOWED_JOB_KINDS": "promotion,unknown"}
        )


def test_r241_container_and_systemd_templates_are_isolated_and_path_preserving() -> None:
    compose = (
        ROOT
        / "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template"
    ).read_text(encoding="utf-8")
    unit = (
        ROOT
        / "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template"
    ).read_text(encoding="utf-8")
    assert "container_name: pokebot-r241-elmo-official-r236-worker-8767" in compose
    assert '"8767:8767"' in compose
    assert "192.168.1.143:8765" not in compose
    assert "192.168.1.158:8766" not in compose
    assert "${R241_ELMO_CHECKPOINT_TRANSPORT_ROOT:?set the Elmo checkpoint transport staging root}:/workspace/checkpoint:ro" in compose
    assert "POKEBOT_REMOTE_CHECKPOINT_ROOT: /workspace/checkpoint" in compose
    for forbidden in (
        "POKEBOT_LIBCG_PATH",
        "POKEBOT_BATCH_LIBCG",
        "POKEBOT_MCTS_",
        "POKEBOT_RTP_",
    ):
        assert forbidden not in compose
    assert "PURE_RL_PUBLIC_MIX_LOCAL_ONLY: \"0\"" in compose
    assert "POKEBOT_BASELINES_DIR:" in compose
    assert "R241_ELMO_BASELINES_ROOT" in compose
    assert "R241_ELMO_H10_BASELINE_ROOT" not in compose
    assert "POKEBOT_REMOTE_ALLOWED_JOB_KINDS: play,self_play,self_play_multi,runtime_probe" in compose
    assert "R241_ELMO_LEARNER_TREE_PARENT" not in compose
    assert "R241_ELMO_CHECKPOINT_PARENT" not in compose
    assert "R241_ELMO_CHECKPOINT_TRANSPORT_STAGING_RECEIPT" in compose
    assert "R241_ELMO_ACTIVATION_OVERLAY" in compose
    expected_adapter = (
        "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        "marnie-h10-direct-policy-adapter-r251-v8.json"
    )
    assert compose.count(expected_adapter) == 2
    assert "runtime/marnie-h10-direct-policy-adapter.json" not in compose
    assert "r241-elmo-official-r236-remote-worker.env" in unit
    assert "pokebot-r241-elmo-official-r236-worker" in unit
    assert "R241_ELMO_SOURCE_SNAPSHOT_ROOT/deploy/elmo" in unit
    assert "/mnt/Main/main/poke-bot-agent/deploy/elmo" not in unit
    assert "8765" not in unit
    assert "8766" not in unit


def test_r241_checkpoint_transport_round_trips_the_generic_remote_path_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _file(tmp_path / "trainer" / "checkpoint.pt", "r241 bytes")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_name = remote_jobs.digest_addressed_basename(checkpoint, digest=digest)
    observed: dict[str, object] = {}

    class _Client:
        def __init__(self, host: str, port: int, **_kwargs: object) -> None:
            observed["host"] = host
            observed["port"] = port

        def connect(self):
            return type("Info", (), {"capabilities": ("checkpoint_digest_verify_v1",)})()

        def verify_checkpoint(self, path: str) -> dict[str, object]:
            observed["path"] = path
            return {"checkpoint_digest": digest}

        def close(self) -> None:
            return None

    monkeypatch.setenv(remote_jobs.ELMO_CHECKPOINT_VERIFY_PORT_ENV, "8767")
    monkeypatch.setattr(remote_jobs, "RemoteJobClient", _Client)
    remote_path = f"/workspace/checkpoint/{expected_name}"
    assert remote_jobs._elmo_remote_checkpoint_digest("192.168.1.143", remote_path) == digest
    assert observed == {
        "host": "192.168.1.143",
        "port": 8767,
        "path": remote_path,
    }

    root = tmp_path / "checkpoint-transport"
    root.mkdir()
    mounted = _file(root / expected_name, "r241 bytes")
    assert run_remote_worker._resolve_checkpoint_within_root(
        mounted, checkpoint_root=root
    ) == mounted.resolve()
    with pytest.raises(ValueError):
        run_remote_worker._resolve_checkpoint_within_root(
            checkpoint, checkpoint_root=root
        )


def test_r241_serve_overlay_binds_existing_preflight_receipts_without_rewriting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(worker, "ELMO_OUTPUTS_ROOT", outputs)
    preflight = _receipt_preflight(tmp_path)
    source_staging = _file(outputs / "source-staging.json", "{}")
    preflight.sources["source_staging_receipt"] = {
        "staging_receipt": str(source_staging),
        "staging_receipt_sha256": worker._sha256(source_staging),
    }
    preflight.sources["runtime_registry"] = {
        "path": "state/alakazam-new-list-direct-r241-runtime-registry.json",
        "sha256": "sha256:" + "d" * 64,
    }
    receipt_dir = tmp_path / "receipts"
    paths = worker.write_r241_elmo_preflight_receipts(
        worker.build_r241_elmo_preflight_receipts(preflight, receipt_dir=receipt_dir),
        receipt_dir=receipt_dir,
    )
    manifest = worker.validate_r241_elmo_preflight_manifest(paths["manifest"])
    baseline = worker._baseline_payload_identity(preflight.baseline_payload)
    source_snapshot = preflight.sources["source_snapshot"]
    authorization = _readonly_json(
        outputs / "owner-start-authorization.json",
        {
            "schema": worker.R241_OWNER_START_AUTHORIZATION_SCHEMA,
            "revision": 241,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "authorized",
            "authorized": True,
            "owner_contract_sha256": preflight.sources["owner_contract"]["sha256"],
            "allowed_actions": ["managed_r241_training_start"],
            "source_snapshot_manifest_sha256": source_snapshot["manifest_sha256"],
            "source_tree_sha256": source_snapshot["source_tree_sha256"],
            "canonical_baseline_manifest_sha256": baseline["baseline_manifest_sha256"],
            "canonical_baseline_roster_sha256": baseline["baseline_roster_sha256"],
            "submission_boundary": {
                "exact_count": 1,
                "checkpoint_source": "expert_before_iter_00010.pt",
                "intermediate_iteration_5_submission_allowed": False,
                "retry_copy_or_duplicate_allowed": False,
            },
            "authorization_provenance": {
                "schema": worker.R241_OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA,
                "create_only": True,
                "explicit_operator_intent": "authorize_managed_r241_training_start",
            },
        },
    )
    receipt_refs = manifest["payload"]["receipts"]
    worker_image_receipt = _readonly_json(
        outputs / "r241-worker-image.json",
        {
            "schema": worker.R241_ELMO_WORKER_IMAGE_SCHEMA,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "sealed_noncanonical_no_network_smoke_passed",
            "create_only": True,
            "image": {
                "tag": "pokebot-r241-elmo-official-r236-worker:test",
                "image_id_sha256": "sha256:" + "e" * 64,
            },
            "source_snapshot": {
                "owner_contract_sha256": preflight.sources["owner_contract"]["sha256"],
                "manifest_sha256": source_snapshot["manifest_sha256"],
                "source_tree_sha256": source_snapshot["source_tree_sha256"],
            },
            "activation_authority": {
                "external_activation_overlay_created": False,
                "managed_service_start_authorized": False,
                "listener_started": False,
                "training_started": False,
            },
            "noncanonical_network_disabled_one_shot_smoke": {
                "validated_external_d162": True,
                "simulator_battles_started": 0,
                "native_function_calls": 0,
                "search_calls": 0,
            },
        },
    )
    overlay_payload = {
        "schema": worker.R241_ACTIVATION_OVERLAY_SCHEMA,
        "revision": 241,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "ready",
        "passed": True,
        "owner_contract_sha256": preflight.sources["owner_contract"]["sha256"],
        "base_registry": dict(preflight.sources["runtime_registry"]),
        "source_snapshot": {
            "schema": worker.R241_SOURCE_SNAPSHOT_SCHEMA,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "owner_contract_sha256": preflight.sources["owner_contract"]["sha256"],
            "status": "ready",
            "manifest_sha256": source_snapshot["manifest_sha256"],
            "source_tree_sha256": source_snapshot["source_tree_sha256"],
            "hosts": {
                "inzi": {
                    "root": "/home/inzi/r241-source",
                    "manifest": "/home/inzi/r241-source/r241-source-snapshot-manifest.json",
                    "outputs_root": "/home/inzi/poke-bot-agent/outputs",
                    "staging_receipt": "/home/inzi/poke-bot-agent/outputs/source-staging.json",
                    "staging_receipt_sha256": "sha256:" + "2" * 64,
                },
                "elmo": {
                    "root": str(preflight.repo_root),
                    "manifest": str(preflight.source_snapshot_manifest),
                    "outputs_root": str(outputs),
                    "staging_receipt": str(source_staging),
                    "staging_receipt_sha256": worker._sha256(source_staging),
                },
            },
        },
        "baseline_payloads": {
            "schema": worker.R241_BASELINE_PAYLOAD_REGISTRY_SCHEMA,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "ready",
            "separately_mounted_and_receipted": True,
            "source_snapshot_fallback_allowed": False,
            "canonical_roster_receipt_sha256": baseline["canonical_roster_receipt_sha256"],
            "canonical_baseline_manifest_sha256": baseline["baseline_manifest_sha256"],
            "canonical_baseline_roster_sha256": baseline["baseline_roster_sha256"],
            "hosts": {
                "inzi": {
                    "root": "/home/inzi/r241-baselines",
                    "manifest": "/home/inzi/r241-baselines/r241-baseline-payload-manifest.json",
                    "manifest_sha256": "sha256:" + "3" * 64,
                    "baseline_tree_sha256": "sha256:" + "4" * 64,
                    "staging_receipt": "/home/inzi/poke-bot-agent/outputs/baseline-staging.json",
                    "staging_receipt_sha256": "sha256:" + "5" * 64,
                    "canonical_roster_receipt": (
                        "/home/inzi/poke-bot-agent/outputs/canonical-baseline-roster.json"
                    ),
                },
                "elmo": {
                    key: baseline[key]
                    for key in (
                        "root",
                        "manifest",
                        "manifest_sha256",
                        "baseline_tree_sha256",
                        "staging_receipt",
                        "staging_receipt_sha256",
                        "canonical_roster_receipt",
                    )
                },
            },
        },
        "peak_r195_preservation": {
            "receipt_sha256_inzi": "sha256:" + "6" * 64,
            "receipt_sha256_elmo": "sha256:" + "7" * 64,
        },
        "remote_collection": {
            "endpoint_id": worker.ELMO_R241_ENDPOINT_ID,
            "manifest_sha256": manifest["sha256"],
            "host_receipt_sha256": receipt_refs["host"]["sha256"],
            "runtime_receipt_sha256": receipt_refs["runtime"]["sha256"],
            "gameplay_receipt_sha256": receipt_refs["gameplay"]["sha256"],
            "checkpoint_transport": {
                "status": "ready",
                **worker._checkpoint_transport_identity(preflight.checkpoint_transport),
                "trainer_visible_root": "/home/inzi/r241-checkpoint-transport",
            },
        },
        "worker_image": {
            "schema": worker.R241_ELMO_WORKER_IMAGE_SCHEMA,
            "image_id_sha256": "sha256:" + "e" * 64,
            "receipt": {
                "path": str(worker_image_receipt),
                "sha256": worker._sha256(worker_image_receipt),
            },
            "source_snapshot": {
                "owner_contract_sha256": preflight.sources["owner_contract"]["sha256"],
                "manifest_sha256": source_snapshot["manifest_sha256"],
                "source_tree_sha256": source_snapshot["source_tree_sha256"],
            },
            "tag": "pokebot-r241-elmo-official-r236-worker:test",
        },
        "mirrors": {
            "schema": "poke_bot.alakazam_new_list_direct_r241_activation_overlay_mirrors/v1",
            "hosts": ["inzi", "elmo"],
            "byte_identical_required": True,
        },
        "owner_start_authorization": {
            "schema": worker.R241_OWNER_START_AUTHORIZATION_SCHEMA,
            "sha256": worker._sha256(authorization),
            "byte_identical_mirrors_required": True,
            "hosts": {
                "inzi": {"path": "/home/inzi/poke-bot-agent/outputs/owner-start-authorization.json"},
                "elmo": {"path": str(authorization)},
            },
        },
    }
    overlay = _readonly_json(outputs / "activation-overlay.json", overlay_payload)
    monkeypatch.setenv(worker.ELMO_R241_WORKER_IMAGE_ID_ENV, "sha256:" + "e" * 64)

    baseline_overlay = overlay_payload["baseline_payloads"]
    assert "canonical_roster_receipt" not in baseline_overlay
    assert baseline_overlay["hosts"]["elmo"]["canonical_roster_receipt"] == baseline[
        "canonical_roster_receipt"
    ]
    assert (
        baseline_overlay["hosts"]["inzi"]["canonical_roster_receipt"]
        != baseline_overlay["hosts"]["elmo"]["canonical_roster_receipt"]
    )

    validated = worker.validate_r241_elmo_activation_overlay(
        overlay_path=overlay,
        overlay_sha256=worker._sha256(overlay),
        preflight=preflight,
        manifest=manifest,
    )
    assert validated["sha256"] == worker._sha256(overlay)
    assert validated["owner_start_authorization_path"] == str(authorization)
    assert validated["worker_image"] == {
        "image_id_sha256": "sha256:" + "e" * 64,
        "receipt_path": str(worker_image_receipt),
        "receipt_sha256": worker._sha256(worker_image_receipt),
    }

    for name, mutate in (
        (
            "missing-elmo-canonical",
            lambda payload: payload["baseline_payloads"]["hosts"]["elmo"].pop(
                "canonical_roster_receipt"
            ),
        ),
        (
            "wrong-elmo-canonical",
            lambda payload: payload["baseline_payloads"]["hosts"]["elmo"].update(
                {
                    "canonical_roster_receipt": (
                        "/mnt/Main/main/poke-bot-agent/outputs/state/wrong-canonical.json"
                    )
                }
            ),
        ),
        (
            "cross-host-canonical",
            lambda payload: payload["baseline_payloads"]["hosts"]["elmo"].update(
                {
                    "canonical_roster_receipt": payload["baseline_payloads"]["hosts"][
                        "inzi"
                    ]["canonical_roster_receipt"]
                }
            ),
        ),
        (
            "unscoped-canonical",
            lambda payload: payload["baseline_payloads"].update(
                {
                    "canonical_roster_receipt": baseline[
                        "canonical_roster_receipt"
                    ]
                }
            ),
        ),
        (
            "null-unscoped-canonical",
            lambda payload: payload["baseline_payloads"].update(
                {"canonical_roster_receipt": None}
            ),
        ),
    ):
        rejected = json.loads(json.dumps(overlay_payload))
        mutate(rejected)
        rejected_overlay = _readonly_json(
            outputs / f"activation-overlay-{name}.json", rejected
        )
        with pytest.raises(
            worker.R241ElmoRemoteWorkerError,
            match="activation overlay baseline payload",
        ):
            worker.validate_r241_elmo_activation_overlay(
                overlay_path=rejected_overlay,
                overlay_sha256=worker._sha256(rejected_overlay),
                preflight=preflight,
                manifest=manifest,
            )

    mismatched = json.loads(json.dumps(overlay_payload))
    mismatched["worker_image"]["source_snapshot"]["source_tree_sha256"] = (
        "sha256:" + "f" * 64
    )
    mismatched_overlay = _readonly_json(
        outputs / "activation-overlay-image-source-mismatch.json", mismatched
    )
    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="worker image does not bind this source snapshot",
    ):
        worker.validate_r241_elmo_activation_overlay(
            overlay_path=mismatched_overlay,
            overlay_sha256=worker._sha256(mismatched_overlay),
            preflight=preflight,
            manifest=manifest,
        )

    monkeypatch.setenv(worker.ELMO_R241_WORKER_IMAGE_ID_ENV, "sha256:" + "f" * 64)
    with pytest.raises(
        worker.R241ElmoRemoteWorkerError,
        match="container image ID does not match activation overlay",
    ):
        worker.validate_r241_elmo_activation_overlay(
            overlay_path=overlay,
            overlay_sha256=worker._sha256(overlay),
            preflight=preflight,
            manifest=manifest,
        )


def test_elmo_image_template_requires_content_id_and_receipt_before_compose() -> None:
    env_template = (
        ROOT / "deploy/elmo/r241-elmo-official-r236-remote-worker.env.template"
    ).read_text(encoding="utf-8")
    unit_template = (
        ROOT
        / "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template"
    ).read_text(encoding="utf-8")
    compose_template = (
        ROOT
        / "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template"
    ).read_text(encoding="utf-8")
    assert "R241_ELMO_OFFICIAL_R236_IMAGE=REPLACE_WITH_SEALED_R241_IMAGE_ID" in env_template
    assert "R241_ELMO_OFFICIAL_R236_IMAGE_ID=REPLACE_WITH_SEALED_R241_IMAGE_ID" in env_template
    assert "R241_ELMO_OFFICIAL_R236_IMAGE_RECEIPT=" in env_template
    assert "R241_ELMO_OFFICIAL_R236_IMAGE_RECEIPT_SHA256=" in env_template
    assert 'image: ${R241_ELMO_OFFICIAL_R236_IMAGE:' in compose_template
    assert "R241_ELMO_OFFICIAL_R236_IMAGE_ID: ${R241_ELMO_OFFICIAL_R236_IMAGE_ID:" in compose_template
    assert 'test "$R241_ELMO_OFFICIAL_R236_IMAGE" = "$R241_ELMO_OFFICIAL_R236_IMAGE_ID"' in unit_template
    assert "/usr/bin/docker image inspect" in unit_template
    assert "R241_ELMO_OFFICIAL_R236_IMAGE_RECEIPT_SHA256" in unit_template
