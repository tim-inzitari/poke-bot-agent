from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/launch_alakazam_new_list_direct_r241.py"
GENERIC_LAUNCHER = ROOT / "scripts/launch_pure_rl.py"


def _module():
    spec = importlib.util.spec_from_file_location("r241_launcher_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generic_launcher_module():
    spec = importlib.util.spec_from_file_location(
        "r241_generic_launcher_test", GENERIC_LAUNCHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r241_static_registry_binds_r251_owner_and_exact_public_plan() -> None:
    launcher = _module()
    _, registry = launcher.load_registry()

    source = launcher.validate_static_registry(registry)
    assert source["latest_owner_clarification_revision"] == 251
    assert registry["owner_contract"]["sha256"] == (
        "sha256:2f9ca8fc0d4cb2a7c6acbc12ecce3e96143a2c9e318e198276ea0dd66bb30c7d"
    )
    assert registry["run"] == {
        "name": "alakazam_new_list_direct_policy_r241",
        "inzi_root": "/home/inzi/poke-bot-agent/outputs/pure_rl/alakazam_new_list_direct_policy_r241",
        "elmo_root": "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/alakazam_new_list_direct_policy_r241",
        "external_activation_overlay_required": True,
        "activation_overlay_schema": launcher.ACTIVATION_OVERLAY_SCHEMA,
        "managed_service_start_authorized": False,
        "submission_authorized": False,
    }
    direct = registry["direct_policy"]
    assert direct["adapter_receipt_inzi"] == (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        "marnie-h10-direct-policy-adapter-r251-v8.json"
    )
    assert direct["adapter_receipt_elmo"] == (
        "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        "marnie-h10-direct-policy-adapter-r251-v8.json"
    )
    preservation = registry["peak_r195_preservation"]
    assert preservation["receipt_inzi"] == (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        "peak-r195-preservation-v6.json"
    )
    assert preservation["receipt_elmo"] == (
        "/mnt/Main/main/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        "peak-r195-preservation-v6.json"
    )
    refresh = registry["matchup_archetype_refresh"]
    assert refresh["status"] == "deferred_not_required"
    assert refresh["required_for_r241_activation"] is False
    assert refresh["slot_change_status"] == "no_slot_change"
    assert refresh["immutable_existing_slot_prefix_count"] == 20
    assert refresh["new_slots"] == []
    assert not any("ptcgreplay" in key.lower() for key in refresh)
    baselines = registry["baseline_payloads"]
    assert baselines["status"] == "pending_external_baseline_payload_snapshot"
    assert baselines["source_snapshot_fallback_allowed"] is False
    assert all(not any(row.values()) for row in baselines["hosts"].values())
    assert {
        "scripts/stage_r241_source_snapshot.py",
        "scripts/stage_r241_elmo_checkpoint_transport.py",
        "scripts/transfer_r241_exact20_alakazam_corpus.py",
        "scripts/canary_game_accuracy.py",
        "scripts/resource_watcher.py",
        "scripts/unattended_monitor.py",
        "scripts/publish_r241_activation_overlay.py",
        "scripts/generate_r241_owner_start_authorization.py",
        "scripts/generate_r241_marnie_direct_policy_adapter_receipt.py",
        "scripts/install_r241_activation_overlay_mirror.py",
        "scripts/generate_r241_canonical_baseline_roster.py",
        "scripts/preflight_alakazam_new_list_direct_r241_service_chain.py",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-finalize.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-submission-queue.service.template",
        "deploy/systemd/pokebot-alakazam-new-list-direct-r241-upload.service.template",
        "deploy/elmo/docker-compose.r241-elmo-official-r236-remote-worker.yml.template",
        "deploy/elmo/r241-elmo-official-r236-remote-worker.env.template",
        "deploy/systemd/pokebot-r241-elmo-official-r236-remote-worker.service.template",
        "poke_bot/r241_canonical_baseline_roster.py",
        "state/alakazam-own-deck-ledger-successor-r258.json",
    }.issubset(launcher._REQUIRED_SOURCE_SNAPSHOT_FILES)
    assert "ops/elmo/run_r241_exact20_specialist_finalizer.sh" not in (
        launcher._REQUIRED_SOURCE_SNAPSHOT_FILES
    )

    assert launcher.planned_collection_group_counts(
        games_per_iteration=8196,
        self_play_fraction=float("0.12493899463152758"),
        strong_public_fraction_of_public=0.50,
        research_control_games=1000,
    ) == {
        "self_play": 1024,
        "strong_public_practice": 4586,
        "diverse_public": 2586,
    }


@pytest.mark.parametrize(
    "inactive_filename",
    (
        "marnie-h10-direct-policy-adapter-r251.json",
        "marnie-h10-direct-policy-adapter-r251-v2.json",
        "marnie-h10-direct-policy-adapter-r251-v3.json",
        "marnie-h10-direct-policy-adapter-r251-v4.json",
        "marnie-h10-direct-policy-adapter-r251-v5.json",
        "marnie-h10-direct-policy-adapter-r251-v6.json",
        "marnie-h10-direct-policy-adapter-r251-v7.json",
    ),
)
def test_r241_static_registry_rejects_the_inactive_h10_receipt_lineage(
    inactive_filename: str,
) -> None:
    launcher = _module()
    _, registry = launcher.load_registry()
    registry = json.loads(json.dumps(registry))
    registry["direct_policy"]["adapter_receipt_inzi"] = (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        + inactive_filename
    )

    with pytest.raises(
        launcher.R241LaunchError,
        match="predeclared successor path for inzi",
    ):
        launcher.validate_static_registry(registry)


@pytest.mark.parametrize(
    "inactive_filename",
    (
        "peak-r195-preservation.json",
        "peak-r195-preservation-v2.json",
        "peak-r195-preservation-v3.json",
        "peak-r195-preservation-v4.json",
        "peak-r195-preservation-v5.json",
    ),
)
def test_r241_static_registry_rejects_the_inactive_peak_receipt_lineage(
    inactive_filename: str,
) -> None:
    launcher = _module()
    _, registry = launcher.load_registry()
    registry = json.loads(json.dumps(registry))
    registry["peak_r195_preservation"]["receipt_inzi"] = (
        "/home/inzi/poke-bot-agent/outputs/pure_rl/"
        "alakazam_new_list_direct_policy_r241/runtime/"
        + inactive_filename
    )

    with pytest.raises(
        launcher.R241LaunchError,
        match="predeclared successor path for inzi",
    ):
        launcher.validate_static_registry(registry)


class _PreservationParentAccepted(Exception):
    """Sentinel proving the preservation parser advanced past its parent."""


def _preservation_parent_fixture(
    launcher: object,
    tmp_path: Path,
    *,
    host_name: str = "inzi",
    local_parent_host: str = "elmo",
) -> tuple[dict[str, object], object, Path, dict[str, object]]:
    parent_bytes = b"immutable-r195-parent"
    registry_parent_checkpoint = tmp_path / "inzi-expert-before-iter-00021.pt"
    registry_parent_checkpoint.write_bytes(parent_bytes)
    parent_checkpoint = (
        tmp_path / f"{local_parent_host}-expert-before-iter-00021.pt"
    )
    parent_checkpoint.write_bytes(parent_bytes)
    parent = launcher.checkpoint_receipts.file_identity(
        parent_checkpoint, label="peak-v6 parent fixture"
    ).as_dict()
    receipt_path = tmp_path / "peak-r195-preservation-v6.json"
    receipt = {
        "schema": launcher.PRESERVATION_SCHEMA,
        "revision": launcher.R241_REVISION,
        "status": "passed",
        "passed": True,
        "derived_not_self_asserted": True,
        "parent": parent,
        "matchup_adapter": {},
    }
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry: dict[str, object] = {
        "peak_r195_preservation": {
            "receipt_schema": launcher.PRESERVATION_SCHEMA,
            f"receipt_sha256_{host_name}": launcher._sha256(receipt_path),
            "learner_matchup_tree_sha256": "sha256:" + "a" * 64,
        },
        "parent": {
            "checkpoint": str(registry_parent_checkpoint),
            "sha256": parent["sha256"],
            "size_bytes": parent["size_bytes"],
        },
    }
    context = launcher.HostContext(
        name=host_name,
        runtime_root=tmp_path / "runtime",
        official_cg_root=tmp_path / "cg-r236",
        adapter_receipt=tmp_path / "marnie-adapter.json",
        preservation_receipt=receipt_path,
        expert_archive_receipt=tmp_path / "expert-current.json",
        expert_manifest_pointer=tmp_path / "PROTECTED_EXPERT_CORPUS.json",
    )
    return registry, context, receipt_path, receipt


def _stop_preservation_after_parent(
    launcher: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "_validate_source_snapshot",
        lambda registry, context, receipt: object(),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_baseline_payload_snapshot",
        lambda registry, context, source_snapshot: object(),
    )

    def accepted(*args: object, **kwargs: object) -> Path:
        raise _PreservationParentAccepted

    monkeypatch.setattr(launcher, "_path_binding", accepted)


@pytest.mark.parametrize(
    ("host_name", "local_parent_host"),
    (("inzi", "inzi-local"), ("elmo", "elmo-local")),
)
def test_peak_v6_typed_parent_file_identity_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    host_name: str,
    local_parent_host: str,
) -> None:
    launcher = _module()
    registry, context, _receipt_path, _receipt = _preservation_parent_fixture(
        launcher,
        tmp_path,
        host_name=host_name,
        local_parent_host=local_parent_host,
    )
    assert context.name == host_name
    assert _receipt["parent"]["path"] != registry["parent"]["checkpoint"]
    _stop_preservation_after_parent(launcher, monkeypatch)

    with pytest.raises(_PreservationParentAccepted):
        launcher._validate_preservation_receipt(registry, context, {})


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("non_mapping", "exact typed FileIdentity"),
        ("missing", "exact typed FileIdentity"),
        ("legacy_hybrid", "exact typed FileIdentity"),
        ("wrong_path", "identity drifted"),
        ("wrong_sha256", "does not pin immutable r195"),
        ("wrong_size", "does not pin immutable r195"),
    ),
)
def test_peak_v6_typed_parent_file_identity_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    launcher = _module()
    registry, context, receipt_path, receipt = _preservation_parent_fixture(
        launcher, tmp_path
    )
    parent = dict(receipt["parent"])
    if case == "non_mapping":
        receipt["parent"] = []
    elif case == "missing":
        parent.pop("path")
    elif case == "legacy_hybrid":
        parent["checkpoint"] = parent["path"]
    elif case == "wrong_path":
        wrong_parent = tmp_path / "another-host-parent.pt"
        wrong_parent.write_bytes(b"x" * int(parent["size_bytes"]))
        parent["path"] = str(wrong_parent)
    elif case == "wrong_sha256":
        parent["sha256"] = "sha256:" + "b" * 64
    elif case == "wrong_size":
        parent["size_bytes"] = int(parent["size_bytes"]) + 1
    if case != "non_mapping":
        receipt["parent"] = parent
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry["peak_r195_preservation"]["receipt_sha256_inzi"] = launcher._sha256(
        receipt_path
    )
    _stop_preservation_after_parent(launcher, monkeypatch)

    with pytest.raises(launcher.R241LaunchError, match=message):
        launcher._validate_preservation_receipt(registry, context, {})


def test_launcher_check_accepts_the_actual_peak_v6_parent_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = _module()
    registry, context, _receipt_path, _receipt = _preservation_parent_fixture(
        launcher, tmp_path
    )
    _stop_preservation_after_parent(launcher, monkeypatch)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    overlay = object()
    preservation = object()

    monkeypatch.setattr(
        launcher,
        "load_registry",
        lambda path: (Path(path), registry),
    )
    monkeypatch.setattr(launcher, "validate_static_registry", lambda registry: {})
    monkeypatch.setattr(
        launcher,
        "apply_activation_overlay",
        lambda *args, **kwargs: (registry, overlay),
    )

    def validate_for_check(
        candidate: object,
        *,
        host: str,
        environment: object = None,
        activation_overlay: object = None,
    ) -> tuple[object, object]:
        assert host == "inzi"
        assert activation_overlay is overlay
        with pytest.raises(_PreservationParentAccepted):
            launcher._validate_preservation_receipt(candidate, context, {})
        return context, preservation

    monkeypatch.setattr(launcher, "validate_activation", validate_for_check)
    monkeypatch.setattr(
        launcher,
        "build_command",
        lambda candidate, actual_context, actual_preservation, python: [
            python,
            "-u",
            "/sealed/scripts/launch_pure_rl.py",
        ],
    )

    assert (
        launcher.main(
            [
                "--registry",
                str(registry_path),
                "--host",
                "inzi",
                "--python",
                "/sealed/python",
                "--activation-overlay",
                str(tmp_path / "overlay.json"),
                "--activation-overlay-sha256",
                "sha256:" + "c" * 64,
                "--activation-overlay-mirror-receipt",
                str(tmp_path / "overlay-mirror.json"),
                "--activation-overlay-mirror-receipt-sha256",
                "sha256:" + "d" * 64,
                "--check",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == (
        "/sealed/python -u /sealed/scripts/launch_pure_rl.py"
    )


def test_ready_baseline_contract_requires_host_scoped_canonical_roster_paths() -> None:
    launcher = _module()
    digest = lambda nibble: "sha256:" + nibble * 64
    baseline = {
        "schema": launcher.BASELINE_PAYLOAD_REGISTRY_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "ready",
        "separately_mounted_and_receipted": True,
        "source_snapshot_fallback_allowed": False,
        "canonical_roster_receipt_sha256": digest("a"),
        "canonical_baseline_manifest_sha256": digest("b"),
        "canonical_baseline_roster_sha256": digest("c"),
        "hosts": {
            "inzi": {
                "root": "/home/inzi/poke-bot-agent-deployments/r241-baseline-a",
                "manifest": "/home/inzi/poke-bot-agent-deployments/r241-baseline-a/r241-baseline-payload-manifest.json",
                "manifest_sha256": digest("d"),
                "baseline_tree_sha256": digest("e"),
                "staging_receipt": "/home/inzi/poke-bot-agent/outputs/state/r241-baseline-staging.json",
                "staging_receipt_sha256": digest("f"),
                "canonical_roster_receipt": "/home/inzi/poke-bot-agent/outputs/state/r241-canonical-baseline-roster.json",
            },
            "elmo": {
                "root": "/mnt/Main/main/poke-bot-agent-deployments/r241-baseline-a",
                "manifest": "/mnt/Main/main/poke-bot-agent-deployments/r241-baseline-a/r241-baseline-payload-manifest.json",
                "manifest_sha256": digest("d"),
                "baseline_tree_sha256": digest("e"),
                "staging_receipt": "/mnt/Main/main/poke-bot-agent/outputs/state/r241-baseline-staging.json",
                "staging_receipt_sha256": digest("f"),
                "canonical_roster_receipt": "/mnt/Main/main/poke-bot-agent/outputs/state/r241-canonical-baseline-roster.json",
            },
        },
    }

    launcher._validate_baseline_payload_contract({"baseline_payloads": baseline})

    for case in ("missing", "cross_host", "unscoped_null", "unscoped_empty_mapping"):
        rejected = json.loads(json.dumps(baseline))
        if case == "missing":
            rejected["hosts"]["inzi"].pop("canonical_roster_receipt")
        elif case == "cross_host":
            rejected["hosts"]["elmo"]["canonical_roster_receipt"] = rejected[
                "hosts"
            ]["inzi"]["canonical_roster_receipt"]
        elif case == "unscoped_null":
            rejected["canonical_roster_receipt"] = None
        else:
            rejected["canonical_roster_receipt"] = {}

        with pytest.raises(launcher.R241LaunchError, match="canonical-roster|host-absolute"):
            launcher._validate_baseline_payload_contract(
                {"baseline_payloads": rejected}
            )


@pytest.mark.parametrize(
    "scoped_role",
    (
        "learner",
        "pinned_h10_marnie_opponent",
        "target_generation",
        "terminal_package_and_submission",
    ),
)
def test_r251_rejects_non_direct_scoped_role_without_touching_frozen_public_mix(
    scoped_role: str,
) -> None:
    """The r251 exemption is only for frozen non-H10 public opponents."""

    launcher = _module()
    _, registry = launcher.load_registry()
    source_path = ROOT / registry["owner_contract"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["search_and_planning_exclusion"]["scope"][scoped_role] = (
        "search_allowed"
    )

    with pytest.raises(
        launcher.R241LaunchError,
        match="direct-policy scope must preserve frozen non-H10 public selectors",
    ):
        launcher._validate_scoped_direct_policy_contract(
            source, registry["direct_policy"]
        )


def test_r241_launcher_rejects_h10_receipt_from_a_stale_source_snapshot(
    tmp_path: Path,
) -> None:
    launcher = _module()
    _, registry = launcher.load_registry()
    source_root = tmp_path / "alakazam-new-list-direct-r241-src-active"
    source_root.mkdir()
    source_manifest = source_root / "r241-source-snapshot-manifest.json"
    source_manifest.write_text('{"source":"active"}\n', encoding="utf-8")
    adapter_receipt = tmp_path / "marnie-h10-direct-policy-adapter-r251-v8.json"
    adapter_receipt.write_text(
        json.dumps(
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
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_snapshot = launcher.SourceSnapshotContext(
        root=source_root,
        manifest=source_manifest,
        manifest_sha256=launcher._sha256(source_manifest),
        source_tree_sha256="sha256:" + "e" * 64,
        outputs_root=tmp_path / "outputs",
    )
    context = launcher.HostContext(
        name="inzi",
        runtime_root=tmp_path / "runtime",
        official_cg_root=tmp_path / "cg-r236",
        adapter_receipt=adapter_receipt,
        preservation_receipt=tmp_path / "preservation.json",
        expert_archive_receipt=tmp_path / "expert.json",
        expert_manifest_pointer=tmp_path / "pointer.json",
    )
    preservation = launcher.PreservationContext(
        active_gate_contract=tmp_path / "gate.json",
        frozen_specialist_registry=tmp_path / "frozen.json",
        research_control_registry=tmp_path / "research.json",
        learner_matchup_tree=tmp_path / "learner-tree.json",
        adapter_activation_receipt=tmp_path / "adapter-activation.json",
        expert_manifest_pointer=context.expert_manifest_pointer,
        preservation_receipt_sha256="sha256:" + "1" * 64,
        adapter_receipt_sha256=launcher._sha256(adapter_receipt),
        official_collect_fraction=0.5,
        research_control_games=1000,
        matchup_adapter_epochs_per_update=1,
        trainer_args=(),
        source_snapshot=source_snapshot,
    )

    with pytest.raises(
        launcher.R241LaunchError,
        match="does not bind the active source snapshot",
    ):
        launcher._validate_marnie_adapter_receipt(
            registry,
            context,
            preservation,
            {},
        )


def test_r241_environment_strips_search_but_keeps_matchup_adapter_runtime() -> None:
    launcher = _module()
    _, registry = launcher.load_registry()
    transport = registry["remote_collection"]["checkpoint_transport"]
    transport.update(
        {
            "status": "ready",
            "host_root": "/host/elmo-checkpoint-transport",
            "trainer_visible_root": "/host/inzi-checkpoint-transport",
            "staging_receipt": "/host/elmo-checkpoint-transport-staging.json",
            "staging_receipt_sha256": "sha256:" + "c" * 64,
            "initial_checkpoint": {
                "container_path": "/workspace/checkpoint/model."
                + "d" * 16
                + ".pt",
                "sha256": "sha256:" + "d" * 64,
            },
        }
    )
    context = launcher.HostContext(
        name="inzi",
        runtime_root=Path("/host/runtime"),
        official_cg_root=Path("/host/cg-r236"),
        adapter_receipt=Path("/host/marnie-adapter.json"),
        preservation_receipt=Path("/host/preservation.json"),
        expert_archive_receipt=Path("/host/expert-current.json"),
        expert_manifest_pointer=Path("/host/PROTECTED_EXPERT_CORPUS.json"),
    )
    source_snapshot = launcher.SourceSnapshotContext(
        root=Path("/host/alakazam-new-list-direct-r241-src-abcdef0123456789"),
        manifest=Path(
            "/host/alakazam-new-list-direct-r241-src-abcdef0123456789/"
            "r241-source-snapshot-manifest.json"
        ),
        manifest_sha256="sha256:" + "2" * 64,
        source_tree_sha256="sha256:" + "3" * 64,
        outputs_root=Path("/host/external-outputs"),
    )
    baseline_payload = launcher.BaselinePayloadContext(
        root=Path("/host/external-baselines"),
        manifest=Path("/host/external-baselines/r241-baseline-payload-manifest.json"),
        manifest_sha256="sha256:" + "5" * 64,
        baseline_tree_sha256="sha256:" + "6" * 64,
        canonical_roster_receipt=Path("/host/canonical-baseline-roster.json"),
        canonical_roster_receipt_sha256="sha256:" + "7" * 64,
        baseline_manifest_sha256="sha256:" + "8" * 64,
        baseline_roster_sha256="sha256:" + "9" * 64,
        baseline_roster=(),
    )
    activation_overlay = launcher.ActivationOverlayContext(
        path=Path("/host/r241-activation-overlay.json"),
        sha256="sha256:" + "a" * 64,
        authorization_receipt=Path("/host/r241-owner-start-authorization.json"),
        authorization_receipt_sha256="sha256:" + "b" * 64,
        mirror_receipt=Path("/host/r241-activation-overlay-mirror.json"),
        mirror_receipt_sha256="sha256:" + "e" * 64,
    )
    preservation = launcher.PreservationContext(
        active_gate_contract=Path("/host/gate.json"),
        frozen_specialist_registry=Path("/host/frozen.json"),
        research_control_registry=Path("/host/research.json"),
        learner_matchup_tree=Path("/host/r195-tree.json"),
        adapter_activation_receipt=Path("/host/adapter-activation.json"),
        expert_manifest_pointer=context.expert_manifest_pointer,
        preservation_receipt_sha256="sha256:" + "4" * 64,
        adapter_receipt_sha256="sha256:" + "1" * 64,
        official_collect_fraction=0.50,
        research_control_games=1000,
        matchup_adapter_epochs_per_update=1,
        trainer_args=(
            "--archetype-aux-loss-weight",
            "0.05",
            "--opp-hand-loss-weight",
            "0.05",
            "--opp-remainder-loss-weight",
            "0.05",
            "--lethal-threat-loss-weight",
            "0.025",
            "--prize-race-loss-weight",
            "0.025",
            "--setup-board-outcome-loss-weight",
            "0.025",
        ),
        source_snapshot=source_snapshot,
        baseline_payload=baseline_payload,
        activation_overlay=activation_overlay,
    )

    environment = launcher.build_environment(
        registry,
        context,
        preservation=preservation,
        environment={
            "POKEBOT_MCTS_SIMS": "64",
            "POKEBOT_RTP_CHECKPOINT": "/old/rtp.pt",
            "POKEBOT_BELIEF_MCTS": "1",
            "POKEBOT_LIBCG_PATH": "/old/cg",
            "POKEBOT_BATCH_LIBCG": "",
            "POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE": "runtime",
            "PURE_RL_EXPERT_REHEARSAL_FORCE_BEFORE": "5",
            "PURE_RL_EXPERT_REHEARSAL_ONE_TIME_EPOCHS": "2",
            "PURE_RL_CONTINUE_AFTER_GATE": "1",
            "PURE_RL_POPULATION_OWN_MODELS_ONLY": "1",
            "POKEBOT_ELMO_SSH_STAGE": "0",
            "POKEBOT_ELMO_CHECKPOINT_HOST_DIR": "/legacy/checkpoints",
            "POKEBOT_ELMO_CHECKPOINT_VERIFY_PORT": "8765",
            "POKEBOT_TRUENAS_CHECKPOINT_SMB": "/legacy/smb",
        },
    )

    assert environment["CG_LIB_PATH"] == "/host/cg-r236"
    assert environment["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "0"
    assert environment["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] == "1"
    assert environment["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] == "/host/r195-tree.json"
    assert environment["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] == "0"
    assert environment["POKEBOT_COMBO_STATE_ROUTE_SPECIALIST"] == "alakazam"
    assert environment["POKEBOT_COMBO_STATE_ROUTE_CHECKPOINT_DIGEST"].startswith(
        "sha256:261d367e"
    )
    assert environment["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] == "runtime"
    assert environment["PURE_RL_EXPERT_REHEARSAL_EVERY"] == "5"
    assert environment["PURE_RL_EXPERT_REHEARSAL_EPOCHS"] == "5"
    assert environment["PURE_RL_TERMINAL_EXPERT_REHEARSAL"] == "1"
    assert environment["PURE_RL_PUBLIC_MIX_LOCAL_ONLY"] == "0"
    assert environment["PURE_RL_OFFICIAL_COLLECT_FRAC"] == "0.5"
    assert environment["PURE_RL_RESEARCH_CONTROL_GAMES_PER_ITER"] == "1000"
    assert environment["POKEBOT_OUTPUTS_DIR"] == "/host/external-outputs"
    assert environment["POKEBOT_BASELINES_DIR"] == "/host/external-baselines"
    assert environment["POKEBOT_R241_BASELINE_PAYLOAD_MANIFEST"] == str(
        baseline_payload.manifest
    )
    assert environment["POKEBOT_R241_SOURCE_EXECUTION_ROOT"] == str(
        source_snapshot.root
    )
    assert environment["POKEBOT_R241_SOURCE_SNAPSHOT_MANIFEST"] == str(
        source_snapshot.manifest
    )
    assert environment["POKEBOT_R241_ACTIVATION_OVERLAY"] == str(
        activation_overlay.path
    )
    assert environment["POKEBOT_R241_ACTIVATION_OVERLAY_SHA256"] == (
        activation_overlay.sha256
    )
    assert environment["POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT"] == str(
        activation_overlay.mirror_receipt
    )
    assert environment["POKEBOT_R241_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256"] == (
        activation_overlay.mirror_receipt_sha256
    )
    assert environment["POKEBOT_ELMO_SSH_STAGE"] == "1"
    assert environment["POKEBOT_ELMO_CHECKPOINT_HOST_DIR"] == (
        "/host/elmo-checkpoint-transport"
    )
    assert environment["POKEBOT_ELMO_CHECKPOINT_VERIFY_PORT"] == "8767"
    assert environment["POKEBOT_TRUENAS_CHECKPOINT_SMB"] == (
        "/host/inzi-checkpoint-transport"
    )
    assert environment["PURE_RL_REMOTE_WORKER_ENDPOINTS"] == "192.168.1.143:8767"
    assert environment["POKEBOT_REMOTE_WORKER_ENDPOINTS"] == "192.168.1.143:8767"
    assert "8765" not in environment["PURE_RL_REMOTE_WORKER_ENDPOINTS"]
    assert not any(
        key.startswith(("POKEBOT_MCTS_", "POKEBOT_RTP_", "POKEBOT_BELIEF_"))
        for key in environment
    )
    assert "POKEBOT_LIBCG_PATH" not in environment
    assert "POKEBOT_BATCH_LIBCG" not in environment
    assert "PURE_RL_EXPERT_REHEARSAL_FORCE_BEFORE" not in environment
    assert "PURE_RL_EXPERT_REHEARSAL_ONE_TIME_EPOCHS" not in environment
    assert "PURE_RL_CONTINUE_AFTER_GATE" not in environment
    assert "PURE_RL_POPULATION_OWN_MODELS_ONLY" not in environment


def test_r241_command_executes_the_snapshot_but_keeps_outputs_external() -> None:
    launcher = _module()
    _, registry = launcher.load_registry()
    context = launcher.HostContext(
        name="inzi",
        runtime_root=Path("/host/external-outputs/pure_rl/alakazam_new_list_direct_policy_r241/runtime"),
        official_cg_root=Path("/host/cg-r236"),
        adapter_receipt=Path("/host/marnie-adapter.json"),
        preservation_receipt=Path("/host/preservation.json"),
        expert_archive_receipt=Path("/host/expert-current.json"),
        expert_manifest_pointer=Path("/host/PROTECTED_EXPERT_CORPUS.json"),
    )
    source_snapshot = launcher.SourceSnapshotContext(
        root=Path("/host/alakazam-new-list-direct-r241-src-abcdef0123456789"),
        manifest=Path(
            "/host/alakazam-new-list-direct-r241-src-abcdef0123456789/"
            "r241-source-snapshot-manifest.json"
        ),
        manifest_sha256="sha256:" + "2" * 64,
        source_tree_sha256="sha256:" + "3" * 64,
        outputs_root=Path("/host/external-outputs"),
    )
    preservation = launcher.PreservationContext(
        active_gate_contract=Path("/host/gate.json"),
        frozen_specialist_registry=Path("/host/frozen.json"),
        research_control_registry=Path("/host/research.json"),
        learner_matchup_tree=Path("/host/r195-tree.json"),
        adapter_activation_receipt=Path("/host/adapter-activation.json"),
        expert_manifest_pointer=context.expert_manifest_pointer,
        preservation_receipt_sha256="sha256:" + "4" * 64,
        adapter_receipt_sha256="sha256:" + "1" * 64,
        official_collect_fraction=0.50,
        research_control_games=1000,
        matchup_adapter_epochs_per_update=1,
        trainer_args=(
            "--archetype-aux-loss-weight",
            "0.05",
            "--opp-hand-loss-weight",
            "0.05",
            "--opp-remainder-loss-weight",
            "0.05",
            "--lethal-threat-loss-weight",
            "0.025",
            "--prize-race-loss-weight",
            "0.025",
            "--setup-board-outcome-loss-weight",
            "0.025",
        ),
        source_snapshot=source_snapshot,
    )

    command = launcher.build_command(
        registry, context, preservation, python="/host/python"
    )

    assert command[:3] == [
        "/host/python",
        "-u",
        "/host/alakazam-new-list-direct-r241-src-abcdef0123456789/scripts/launch_pure_rl.py",
    ]
    assert "/host/external-outputs/logs/alakazam_new_list_direct_policy_r241.log" in command
    assert "/host/repo" not in command
    assert "--remote-worker-endpoints" in command
    remote_index = command.index("--remote-worker-endpoints")
    assert command[remote_index + 1] == "192.168.1.143:8767"
    preservation_index = command.index("--r241-peak-r195-preservation-receipt")
    assert command[preservation_index + 1] == "/host/preservation.json"
    preservation_sha_index = command.index(
        "--r241-peak-r195-preservation-receipt-sha256"
    )
    assert command[preservation_sha_index + 1] == "sha256:" + "4" * 64


def test_r241_source_snapshot_inventory_is_full_and_output_root_is_external(
    tmp_path: Path,
) -> None:
    launcher = _module()
    _, base_registry = launcher.load_registry()
    registry = json.loads(json.dumps(base_registry))
    source_root = tmp_path / "alakazam-new-list-direct-r241-src-abcdef0123456789"
    source_root.mkdir()
    outputs_root = tmp_path / "external-outputs"
    outputs_root.mkdir()
    rows = []
    for relative in sorted(launcher._REQUIRED_SOURCE_SNAPSHOT_FILES):
        member = source_root / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(f"snapshot member: {relative}\n", encoding="utf-8")
        member.chmod(0o444)
        rows.append(
            {
                "path": relative,
                "sha256": launcher._sha256(member),
                "size_bytes": member.stat().st_size,
            }
        )
    tree_sha256 = launcher._source_tree_digest(rows)
    manifest = source_root / "r241-source-snapshot-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": launcher.SOURCE_SNAPSHOT_SCHEMA,
                "candidate_id": "alakazam-new-list-direct-policy-r241",
                "owner_contract_sha256": registry["owner_contract"]["sha256"],
                "source_tree_sha256": tree_sha256,
                "external_outputs_required": True,
                "baseline_payloads_separate_and_receipted": True,
                "authenticated": True,
                "status": "authenticated_immutable_source_snapshot",
                "files": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    for directory in sorted(
        (path for path in source_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    source_root.chmod(0o555)
    registry["source_snapshot"] = {
        "schema": launcher.SOURCE_SNAPSHOT_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "owner_contract_sha256": registry["owner_contract"]["sha256"],
        "status": "ready",
        "manifest_sha256": launcher._sha256(manifest),
        "source_tree_sha256": tree_sha256,
        "hosts": {
            "inzi": {
                "root": str(source_root),
                "manifest": str(manifest),
                "outputs_root": str(outputs_root),
            },
            "elmo": {
                "root": "/mnt/Main/main/poke-bot-agent-deployments/alakazam-new-list-direct-r241-src-abcdef0123456789",
                "manifest": "/mnt/Main/main/poke-bot-agent-deployments/alakazam-new-list-direct-r241-src-abcdef0123456789/r241-source-snapshot-manifest.json",
                "outputs_root": "/mnt/Main/main/poke-bot-agent/outputs",
            },
        },
    }
    context = launcher.HostContext(
        name="inzi",
        runtime_root=(
            outputs_root
            / "pure_rl"
            / "alakazam_new_list_direct_policy_r241"
            / "runtime"
        ),
        official_cg_root=tmp_path / "cg-r236",
        adapter_receipt=tmp_path / "adapter.json",
        preservation_receipt=tmp_path / "preservation.json",
        expert_archive_receipt=tmp_path / "expert.json",
        expert_manifest_pointer=tmp_path / "PROTECTED_EXPERT_CORPUS.json",
    )
    receipt = {
        "source_snapshot": {
            "schema": launcher.SOURCE_SNAPSHOT_SCHEMA,
            "host": "inzi",
            "root": str(source_root),
            "source_execution_root": str(source_root),
            "manifest": str(manifest),
            "manifest_sha256": launcher._sha256(manifest),
            "source_tree_sha256": tree_sha256,
            "outputs_root": str(outputs_root),
        }
    }

    snapshot = launcher._validate_source_snapshot(registry, context, receipt)

    assert snapshot.root == source_root.resolve()
    assert snapshot.outputs_root == outputs_root.resolve()
    source_root.chmod(0o755)
    for path in source_root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def test_generic_launcher_honors_r241_external_outputs_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    external_outputs = tmp_path / "external-outputs"
    monkeypatch.setenv("POKEBOT_OUTPUTS_DIR", str(external_outputs))
    module = _generic_launcher_module()

    assert module.OUTPUTS_ROOT == external_outputs.resolve()
    assert module.DEFAULT_LOG == external_outputs.resolve() / "logs/pure_rl.log"
    assert module.LAUNCH_LOCK == external_outputs.resolve() / "state/pure_rl_launcher.lock"
    assert module.progress_log_path(Path("relative.log")) == (
        external_outputs.resolve() / "relative.progress.log"
    )
    armed, arm_file = module._production_training_arm(
        {"POKEBOT_TRAINING_ARM_FILE": "relative-arm"}
    )
    assert armed is False
    assert arm_file == external_outputs.resolve() / "relative-arm"


def test_generic_launcher_rejects_an_incomplete_r241_snapshot_helper_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setattr(
        _generic_launcher_module(),
        "ROOT",
        source_root,
    )
    module = _generic_launcher_module()
    monkeypatch.setattr(module, "ROOT", source_root)

    with pytest.raises(RuntimeError, match="omits required subprocess helper"):
        module._validate_r241_snapshot_subprocess_closure(
            {"POKEBOT_R241_SOURCE_EXECUTION_ROOT": str(source_root)}
        )


def test_r241_activation_requires_the_external_overlay() -> None:
    launcher = _module()
    _, registry = launcher.load_registry()

    with pytest.raises(
        launcher.R241LaunchError,
        match="explicit checksum-bound external activation overlay",
    ):
        launcher.validate_activation(registry, host="inzi", environment={})
