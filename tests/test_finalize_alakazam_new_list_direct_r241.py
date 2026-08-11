from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

from scripts import finalize_alakazam_new_list_direct_r241 as finalizer
from scripts import process_alakazam_new_list_direct_r241_submission_queue as queue_processor
from scripts import upload_alakazam_new_list_direct_r241_submission_queue as uploader


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "digest": finalizer.sha256_file(path),
        "sha256": finalizer.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _fake_sha(seed: str) -> str:
    return finalizer.sha256_bytes(seed.encode("utf-8"))


def _run_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    iter9 = checkpoints / "iter_00009.pt"
    iter9.write_bytes(b"durable-iter-00009-direct-policy")
    terminal = checkpoints / "expert_before_iter_00010.pt"
    terminal.write_bytes(b"terminal-five-epoch-direct-policy")
    terminal_parent = _identity(iter9)
    terminal_output = _identity(terminal)
    expert_staging = json.loads(
        finalizer.EXPERT_WINDOW_STAGING_PATH.read_text(encoding="utf-8")
    )
    expert_window = {
        "staging_receipt": _identity(finalizer.EXPERT_WINDOW_STAGING_PATH),
        "canonical_receipt_sha256": finalizer.EXPERT_WINDOW_CANONICAL_RECEIPT_SHA256,
        "immutable_window_receipt_sha256": expert_staging["immutable_window_receipt"][
            "sha256"
        ],
        "window": {
            "start": "2026-07-22",
            "end": "2026-08-10",
            "days": 20,
            "validated_episodes": 91253,
        },
    }

    history: list[dict[str, object]] = []
    terminal_commit: dict[str, object] | None = None
    for iteration in range(10):
        history.append({"iteration": iteration, "completed": True})
        commit = {
            "last_completed_iteration": iteration,
            "next_iteration": iteration + 1,
            "history": list(history),
            "learner": terminal_parent,
        }
        path = run / "commits" / f"iter_{iteration:05d}.json"
        _write_json(path, commit)
        if iteration == 9:
            terminal_commit = commit
    assert terminal_commit is not None

    rehearsal = {
        "schema": 5,
        "before_iteration": 10,
        "parent_digest": terminal_parent["digest"],
        "checkpoint": terminal_output["path"],
        "checkpoint_digest": terminal_output["digest"],
        "epochs": 5,
        "expert_window": expert_window,
    }
    rehearsal_path = run / "rehearsals" / "before_iter_00010.json"
    _write_json(rehearsal_path, rehearsal)
    refresh = {
        "schema": finalizer.TERMINAL_REFRESH_SCHEMA,
        "before_iteration": 10,
        "rl_updates_completed": 10,
        "epochs_completed": 5,
        "parent": terminal_parent,
        "refreshed": terminal_output,
        "expert_rehearsal": rehearsal,
        "next_collection_started": False,
        "expert_window": expert_window,
    }
    refresh_path = run / "terminal_expert_refresh.json"
    _write_json(refresh_path, refresh)
    loop = {
        **terminal_commit,
        "terminal_expert_refresh": {
            "path": str(refresh_path.resolve()),
            "parent": terminal_parent,
            "refreshed": terminal_output,
            "epochs": 5,
        },
    }
    _write_json(run / "loop_state.json", loop)
    return run, terminal, refresh_path


def _fake_canonical_stage(
    tmp_path: Path,
    contract: dict[str, object],
) -> tuple[Path, Path]:
    native_names = {
        "linux_x86_64": "libcg.so",
        "linux_aarch64": "libcg-arm64.so",
        "macos_arm64": "libcg.dylib",
        "windows_x86_64": "cg.dll",
    }
    native: dict[str, dict[str, object]] = {}
    values: dict[str, bytes] = {}
    for platform, filename in native_names.items():
        body = f"fake-official-r236-{platform}".encode("utf-8")
        values[platform] = body
        native[platform] = {
            "package_relative_path": f"cg/{filename}",
            "sha256": finalizer.sha256_bytes(body),
            "size_bytes": len(body),
        }
    canonical = tmp_path / "canonical-r236.json"
    _write_json(
        canonical,
        {
            "schema": "poke_bot.canonical_libcg_r236/v1",
            "canonical_native_libraries": native,
        },
    )
    simulator = dict(contract["canonical_simulator"])
    simulator["typed_source"] = str(canonical.resolve())
    simulator["typed_source_sha256"] = finalizer.sha256_file(canonical)
    simulator["linux_x86_64_sha256"] = native["linux_x86_64"]["sha256"]
    simulator["linux_x86_64_size_bytes"] = native["linux_x86_64"]["size_bytes"]
    contract["canonical_simulator"] = simulator

    sealed = tmp_path / "sealed-r236-runtime"
    cg = sealed / "cg"
    cg.mkdir(parents=True)
    for name in ("__init__.py", "api.py", "game.py", "sim.py"):
        (cg / name).write_text(f"# sealed {name}\n", encoding="utf-8")
    for platform, member in native.items():
        (sealed / str(member["package_relative_path"])).write_bytes(values[platform])
    # An extra inherited sidecar is deliberately not package-eligible.
    (cg / "legacy_search_config.json").write_text("{}\n", encoding="utf-8")
    preflight = {
        "schema": finalizer.OFFICIAL_LIBCG_PREFLIGHT_SCHEMA,
        "revision": 241,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        "local_only": True,
        "direct_policy_only": True,
        "cg_lib_path": str(sealed.resolve()),
        "canonical_native_members": {
            platform: {
                "path": member["package_relative_path"],
                "sha256": member["sha256"],
                "size_bytes": member["size_bytes"],
            }
            for platform, member in native.items()
        },
        "loaded_library": {
            "target_platform": "linux_x86_64",
            "path": str((sealed / "cg/libcg.so").resolve()),
            "sha256": native["linux_x86_64"]["sha256"],
            "size_bytes": native["linux_x86_64"]["size_bytes"],
        },
        "native_export_attestation": {
            "method": "ctypes_symbol_resolution_only",
            "native_function_calls": 0,
        },
        "environment": {
            "CG_LIB_PATH": str(sealed.resolve()),
            "forbidden_override_keys_absent": True,
            "forbidden_override_keys": [
                "POKEBOT_LIBCG_PATH",
                "POKEBOT_BATCH_LIBCG",
            ],
        },
        "wrapper_source": {
            "copied_member_count": 4,
            "discarded_native_members": {"cg/libcg.so": {"sha256": _fake_sha("old")}},
        },
    }
    _write_json(sealed / finalizer.OFFICIAL_LIBCG_PREFLIGHT_NAME, preflight)
    staging = tmp_path / "r241-official-staging.json"
    _write_json(
        staging,
        {
            "schema": finalizer.OFFICIAL_LIBCG_STAGING_SCHEMA,
            "owner_decision_revision": 241,
            "status": "staged_and_export_attested_on_inzi_and_elmo",
            "source": {
                "canonical_contract": str(canonical.resolve()),
                "canonical_contract_sha256": finalizer.sha256_file(canonical),
                "wheel_sha256": simulator["wheel_sha256"],
            },
            "required_runtime": {
                "environment": "CG_LIB_PATH",
                "member": "cg/libcg.so",
                "member_sha256": native["linux_x86_64"]["sha256"],
                "member_size_bytes": native["linux_x86_64"]["size_bytes"],
                "forbidden_environment_absent": [
                    "POKEBOT_LIBCG_PATH",
                    "POKEBOT_BATCH_LIBCG",
                ],
                "native_function_calls_during_export_attestation": 0,
                "search_calls_during_export_attestation": 0,
            },
            "hosts": {
                name: {
                    "passed": True,
                    "loaded_member_sha256": native["linux_x86_64"]["sha256"],
                    "runtime_root": f"/{name}/sealed-r236",
                    "receipt": f"/{name}/sealed-r236/{finalizer.OFFICIAL_LIBCG_PREFLIGHT_NAME}",
                    "receipt_sha256": _fake_sha(f"{name}-receipt"),
                }
                for name in ("inzi", "elmo")
            },
            "scope": {
                "runtime_roots_created": True,
                "complete_four_platform_native_set_staged_per_root": True,
                "old_wrapper_native_members_discarded": True,
                "simulator_battles_started": 0,
                "training_or_gradient_updates_started": False,
                "managed_services_started_or_restarted": False,
                "mcts_or_rtp_authority": False,
                "selector_or_submission_authority": False,
            },
        },
    )
    return sealed, staging


def _contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    contract = json.loads(
        (ROOT / "state/alakazam-new-list-direct-policy-r241.json").read_text(
            encoding="utf-8"
        )
    )
    sealed, staging = _fake_canonical_stage(tmp_path, contract)
    path = tmp_path / "r241-contract.json"
    _write_json(path, contract)
    return path, sealed, staging


def _runtime_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "direct-runtime"
    runtime.mkdir()
    (runtime / "main.py").write_text(
        "def agent(observation, configuration):\n    return 0\n", encoding="utf-8"
    )
    (runtime / "direct_policy.py").write_text(
        "def choose_action(logits):\n    return max(range(len(logits)), key=logits.__getitem__)\n",
        encoding="utf-8",
    )
    # Simulate a legacy source tree.  The whole inherited cg/ directory is
    # culled; only the separately sealed official root may be packaged.
    (runtime / "cg").mkdir()
    (runtime / "cg" / "libcg.so").write_bytes(b"legacy-ffd-libcg")
    (runtime / "cg" / "api.py").write_text("legacy = True\n", encoding="utf-8")
    (runtime / "cg" / "search_config.json").write_text("{}\n", encoding="utf-8")
    return runtime


def _activation_fixtures(
    tmp_path: Path,
    terminal: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    tree = tmp_path / "marnie-h10-direct-matchup-tree.json"
    _write_json(
        tree,
        {
            "runtime_enabled": True,
            "runtime_contract": {
                "accepted_archetype_ids": ["alakazam", "marnie", "unknown"],
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        },
    )
    monkeypatch.setattr(
        finalizer, "LEARNER_R195_MATCHUP_TREE_SHA256", finalizer.sha256_file(tree)
    )
    monkeypatch.setattr(finalizer, "H10_DIRECT_MATCHUP_TREE_SHA256", finalizer.sha256_file(tree))
    monkeypatch.setattr(finalizer, "H10_DIRECT_MATCHUP_TREE_SIZE_BYTES", tree.stat().st_size)
    terminal_identity = _identity(terminal)
    tree_identity = _identity(tree)
    inventory = finalizer._load_peak_r195_head_inventory()
    heads = list(inventory.head_names)
    routes = list(inventory.fusion_route_ids)
    slot_migration = {
        "schema": finalizer.checkpoint_receipts.R241_ADAPTER_SLOT_MIGRATION_SCHEMA,
        "status": "no_slot_change",
        "existing_slots_byte_immutable": True,
        "retained_slot_count": 20,
        "new_slots": [],
        "new_slot_proofs": [],
    }
    runtime_smoke = {
        "model_reconstructed": True,
        "adapter_runtime_enabled_for_smoke": True,
        "adapter_output_changed": True,
        "action_selector": "direct_policy_only",
        "mcts_calls": 0,
        "rtp_calls": 0,
        "search_calls": 0,
    }
    matchup = tmp_path / "matchup-runtime-activation.json"
    model = tmp_path / "model-runtime-activation.json"
    model_payload = {
        "schema": finalizer.MODEL_RUNTIME_ACTIVATION_SCHEMA,
        "owner_decision_revision": 241,
        "owner_clarification_revision": finalizer.LATEST_OWNER_CLARIFICATION_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "active_peak_r195_non_combo_fusion",
        "derived_not_self_asserted": True,
        "parent_r195_checkpoint": {"sha256": finalizer.PARENT_R195_SHA256},
        "heads": {
            "architecture_present_head_count": 19,
            "non_combo_head_count": 18,
            "non_combo_route_count": 18,
            "active_non_combo_head_names": heads,
            "active_non_combo_route_names": heads,
            "active_non_combo_fusion_route_ids": routes,
            "every_non_combo_head_trainable": True,
            "every_non_combo_fusion_route_enabled": True,
            "combo_state": {
                "head_present": True,
                "physical_route_present": True,
                "loss_weight": 0.0,
                "route_enabled": False,
            },
        },
        "runtime_package_activation": {
            "matchup_adapters_enabled": True,
            "checkpoint_remains_dormant": True,
        },
        "adapter_slot_migration": slot_migration,
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
        "terminal_checkpoint": terminal_identity,
        "runtime_smoke": runtime_smoke,
    }
    model_payload["receipt_fingerprint_sha256"] = finalizer.checkpoint_receipts.sha256_bytes(
        finalizer.checkpoint_receipts.canonical_json(model_payload)
    )
    _write_json(model, model_payload)
    matchup_payload = {
        "schema": finalizer.MATCHUP_RUNTIME_ACTIVATION_SCHEMA,
        "owner_decision_revision": 241,
        "owner_clarification_revision": finalizer.LATEST_OWNER_CLARIFICATION_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "active_direct_policy_only",
        "derived_not_self_asserted": True,
        "parent_r195_checkpoint": {"sha256": finalizer.PARENT_R195_SHA256},
        "terminal_checkpoint": terminal_identity,
        "learner_matchup_tree": tree_identity,
        "h10_training_opponent": {
            "matchup_tree": tree_identity,
            "direct_policy_only": True,
            "mcts_enabled": False,
            "recursive_turn_planner_enabled": False,
            "search_enabled": False,
        },
        "adapter_slot_migration": slot_migration,
        "runtime_smoke": runtime_smoke,
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
    }
    matchup_payload["receipt_fingerprint_sha256"] = finalizer.checkpoint_receipts.sha256_bytes(
        finalizer.checkpoint_receipts.canonical_json(matchup_payload)
    )
    _write_json(matchup, matchup_payload)
    return tree, matchup, model


def _prepared_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    run, terminal, _refresh = _run_fixture(tmp_path)
    contract, sealed, staging = _contract(tmp_path)
    runtime = _runtime_fixture(tmp_path)
    tree, matchup, model = _activation_fixtures(tmp_path, terminal, monkeypatch)
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_payload["peak_r195_behavior_preservation"][
        "learner_public_matchup_tree_sha256"
    ] = finalizer.LEARNER_R195_MATCHUP_TREE_SHA256
    contract_payload["peak_r195_behavior_preservation"][
        "marnie_public_matchup_tree_sha256"
    ] = finalizer.H10_DIRECT_MATCHUP_TREE_SHA256
    _write_json(contract, contract_payload)
    # These package-focused fixtures intentionally use tiny non-PyTorch files.
    # The checkpoint-derived validator itself has focused coverage in
    # test_r241_checkpoint_receipts; keep the package tests about archive and
    # authorization behavior while a separate test below proves the finalizer
    # cannot bypass that mandatory validator.
    monkeypatch.setattr(
        finalizer,
        "_validate_generated_terminal_checkpoint_receipts",
        lambda **_kwargs: ({}, {}),
    )
    return {
        "run": run,
        "terminal": terminal,
        "contract": contract,
        "sealed": sealed,
        "staging": staging,
        "runtime": runtime,
        "tree": tree,
        "matchup": matchup,
        "model": model,
    }


def _finalize_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, Path]]:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    paths["output"] = tmp_path / "output"
    paths["authorization"] = tmp_path / "queue" / "r241-authorize.json"
    paths["receipt"] = tmp_path / "receipts" / "r241-finalizer.json"
    result = finalizer.finalize_terminal(
        run_dir=paths["run"],
        runtime_dir=paths["runtime"],
        official_cg_dir=paths["sealed"],
        official_libcg_staging_path=paths["staging"],
        output_dir=paths["output"],
        matchup_tree_path=paths["tree"],
        matchup_runtime_activation_path=paths["matchup"],
        model_runtime_activation_path=paths["model"],
        receipt_path=paths["receipt"],
        queue_authorization_path=paths["authorization"],
        contract_path=paths["contract"],
    )
    return result, paths


def _finalize_again(paths: dict[str, Path]) -> dict[str, object]:
    return finalizer.finalize_terminal(
        run_dir=paths["run"],
        runtime_dir=paths["runtime"],
        official_cg_dir=paths["sealed"],
        official_libcg_staging_path=paths["staging"],
        output_dir=paths["output"],
        matchup_tree_path=paths["tree"],
        matchup_runtime_activation_path=paths["matchup"],
        model_runtime_activation_path=paths["model"],
        receipt_path=paths["receipt"],
        queue_authorization_path=paths["authorization"],
        contract_path=paths["contract"],
    )


def test_r241_finalizer_packages_only_sealed_direct_policy_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, paths = _finalize_fixture(tmp_path, monkeypatch)
    archive = paths["output"] / "submission.tar.gz"

    assert result["direct_submission_performed"] is False
    assert result["network_io_performed"] is False
    assert result["queue_authorizations_emitted"] == 1
    assert archive.is_file()
    assert archive.stat().st_size <= finalizer.ARCHIVE_MAX_BYTES

    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    assert authorization["submission_count_authorized"] == 1
    assert authorization["submission_count_emitted"] == 1
    assert authorization["turn_order_preference"] == "first_if_allowed"
    assert authorization["direct_submission_performed"] is False
    assert len(authorization["queue_entries"]) == 1
    entry = authorization["queue_entries"][0]
    assert entry["sequence"] == 1
    assert entry["remaining_uses"] == 1
    assert entry["retry_allowed"] is False
    assert entry["duplicate_allowed"] is False

    with tarfile.open(archive, "r:gz") as bundle:
        members = {
            member.name: bundle.extractfile(member).read()
            for member in bundle.getmembers()
            if member.isfile()
        }
    assert "cg/search_config.json" not in members
    assert "search_config.json" not in members
    assert "belief_decks.json" not in members
    assert members["cg/libcg.so"] != b"legacy-ffd-libcg"
    assert "model_runtime_activation.json" in members
    assert "matchup_runtime_activation.json" in members
    assert set(
        ("cg/libcg.so", "cg/libcg-arm64.so", "cg/libcg.dylib", "cg/cg.dll")
    ).issubset(members)
    profile = json.loads(members["runtime_profile.json"])
    assert profile["action_selector"] == "direct_policy_only"
    assert profile["mcts_enabled"] is False
    assert profile["recursive_turn_planner_enabled"] is False
    assert profile["search_enabled"] is False
    assert profile["belief_assets_enabled"] is False
    manifest = json.loads(members["package_manifest.json"])
    assert "cg/search_config.json" in manifest["culled_runtime_cg_members"]
    assert "legacy_search_config.json" in manifest["culled_official_cg_members"]
    assert manifest["terminal_model_provenance"]["source"] == (
        "terminal_expert_before_iter_00010_five_epoch_receipt"
    )
    assert manifest["model_runtime"]["all_peak_r195_non_combo_heads_active"] is True
    assert manifest["model_runtime"]["fusion_routes_active"] is True
    assert manifest["model_runtime"]["combo_state_route_enabled"] is False
    assert manifest["model_runtime"]["matchup_adapter_runtime_enabled"] is True


def test_r241_finalizer_is_idempotent_and_never_emits_a_second_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, paths = _finalize_fixture(tmp_path, monkeypatch)
    archive = paths["output"] / "submission.tar.gz"
    before_archive = finalizer.sha256_file(archive)
    before_authorization = paths["authorization"].read_bytes()
    before_receipt = paths["receipt"].read_bytes()

    second = _finalize_again(paths)

    assert first["queue_authorizations_emitted"] == second["queue_authorizations_emitted"] == 1
    assert finalizer.sha256_file(archive) == before_archive
    assert paths["authorization"].read_bytes() == before_authorization
    assert paths["receipt"].read_bytes() == before_receipt


def test_r241_finalizer_rejects_a_second_authorization_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _first, paths = _finalize_fixture(tmp_path, monkeypatch)

    with pytest.raises(finalizer.R241FinalizerError, match="authorization binding"):
        finalizer.finalize_terminal(
            run_dir=paths["run"],
            runtime_dir=paths["runtime"],
            official_cg_dir=paths["sealed"],
            official_libcg_staging_path=paths["staging"],
            output_dir=tmp_path / "another-output",
            matchup_tree_path=paths["tree"],
            matchup_runtime_activation_path=paths["matchup"],
            model_runtime_activation_path=paths["model"],
            receipt_path=tmp_path / "another-receipt.json",
            queue_authorization_path=tmp_path / "another-authorization.json",
            contract_path=paths["contract"],
        )


def test_r241_finalizer_rejects_nonterminal_epoch_count(tmp_path: Path) -> None:
    run, _terminal, refresh_path = _run_fixture(tmp_path)
    contract, _sealed, _staging = _contract(tmp_path)
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    refresh["epochs_completed"] = 4
    _write_json(refresh_path, refresh)

    with pytest.raises(finalizer.R241FinalizerError, match="5-epoch"):
        finalizer.validate_terminal_evidence(run_dir=run, contract_path=contract)


def test_r241_finalizer_cannot_bypass_checkpoint_derived_runtime_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)

    def reject_direct_checkpoint_audit(**_kwargs: object) -> tuple[dict, dict]:
        raise finalizer.R241FinalizerError("derived terminal receipt rejected")

    monkeypatch.setattr(
        finalizer,
        "_validate_generated_terminal_checkpoint_receipts",
        reject_direct_checkpoint_audit,
    )
    with pytest.raises(finalizer.R241FinalizerError, match="derived terminal receipt"):
        finalizer.finalize_terminal(
            run_dir=paths["run"],
            runtime_dir=paths["runtime"],
            official_cg_dir=paths["sealed"],
            official_libcg_staging_path=paths["staging"],
            output_dir=tmp_path / "output",
            matchup_tree_path=paths["tree"],
            matchup_runtime_activation_path=paths["matchup"],
            model_runtime_activation_path=paths["model"],
            receipt_path=tmp_path / "receipt.json",
            queue_authorization_path=tmp_path / "authorization.json",
            contract_path=paths["contract"],
        )


def test_r241_finalizer_rejects_any_iter_10_collection_artifact(tmp_path: Path) -> None:
    run, _terminal, _refresh = _run_fixture(tmp_path)
    contract, _sealed, _staging = _contract(tmp_path)
    (run / "collection_receipts").mkdir()
    (run / "collection_receipts" / "iter_00010.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(finalizer.R241FinalizerError, match="iteration 10"):
        finalizer.validate_terminal_evidence(run_dir=run, contract_path=contract)


@pytest.mark.parametrize(
    "forbidden_name",
    ["mcts_policy.py", "rtp_sidecar.pt", "belief_decks.json", "recursive_turn_planner.py"],
)
def test_r241_finalizer_rejects_all_forbidden_planner_asset_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbidden_name: str
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    (paths["runtime"] / forbidden_name).write_text("blocked\n", encoding="utf-8")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    with pytest.raises(finalizer.R241FinalizerError, match="forbidden asset"):
        finalizer.audit_runtime_source(
            runtime_dir=paths["runtime"],
            official_cg_dir=paths["sealed"],
            contract=contract,
            official_libcg_staging_path=paths["staging"],
        )


def test_r241_finalizer_rejects_planner_code_under_an_innocent_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    (paths["runtime"] / "direct_policy.py").write_text(
        "MCTS = object()\n", encoding="utf-8"
    )
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    with pytest.raises(finalizer.R241FinalizerError, match="forbidden planner/belief token"):
        finalizer.audit_runtime_source(
            runtime_dir=paths["runtime"],
            official_cg_dir=paths["sealed"],
            contract=contract,
            official_libcg_staging_path=paths["staging"],
        )


def test_r241_finalizer_rejects_unsealed_runtime_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    (paths["runtime"] / "opaque_sidecar.pyc").write_bytes(b"not-permitted")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    with pytest.raises(finalizer.R241FinalizerError, match="unsealed executable/model sidecar"):
        finalizer.audit_runtime_source(
            runtime_dir=paths["runtime"],
            official_cg_dir=paths["sealed"],
            contract=contract,
            official_libcg_staging_path=paths["staging"],
        )


def test_r241_finalizer_requires_full_peak_r195_model_activation_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    activation = json.loads(paths["model"].read_text(encoding="utf-8"))
    activation["heads"]["active_non_combo_head_names"] = activation["heads"][
        "active_non_combo_head_names"
    ][:-1]
    activation.pop("receipt_fingerprint_sha256")
    activation["receipt_fingerprint_sha256"] = finalizer.checkpoint_receipts.sha256_bytes(
        finalizer.checkpoint_receipts.canonical_json(activation)
    )
    _write_json(paths["model"], activation)

    with pytest.raises(finalizer.R241FinalizerError, match="model runtime inventory"):
        _finalize_again({
            **paths,
            "output": tmp_path / "out",
            "authorization": tmp_path / "authorization.json",
            "receipt": tmp_path / "receipt.json",
        })


def test_r241_finalizer_enforces_exact_197_7_mib_cap_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepared_fixture(tmp_path, monkeypatch)
    paths.update(
        {
            "output": tmp_path / "out",
            "authorization": tmp_path / "queue" / "authorization.json",
            "receipt": tmp_path / "receipt.json",
        }
    )
    monkeypatch.setattr(finalizer, "ARCHIVE_MAX_BYTES", 1)

    with pytest.raises(finalizer.R241FinalizerError, match="197.7 MiB"):
        _finalize_again(paths)
    assert not paths["authorization"].exists()


def test_r241_direct_queue_processor_consumes_only_one_terminal_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, paths = _finalize_fixture(tmp_path, monkeypatch)
    queue = tmp_path / "direct-policy-queue.json"
    preview = queue_processor.enqueue_authorized_handoff(
        authorization_path=paths["authorization"],
        finalizer_receipt_path=paths["receipt"],
        queue_path=queue,
        contract_path=paths["contract"],
        official_libcg_staging_path=paths["staging"],
        enqueue=False,
    )
    assert preview["status"] == "preflight_passed_no_queue_written"
    assert not queue.exists()
    first = queue_processor.enqueue_authorized_handoff(
        authorization_path=paths["authorization"],
        finalizer_receipt_path=paths["receipt"],
        queue_path=queue,
        contract_path=paths["contract"],
        official_libcg_staging_path=paths["staging"],
        enqueue=True,
    )
    assert first["status"] == "enqueued_pending_explicit_direct_policy_uploader"
    assert first["direct_submission_performed"] is False
    assert first["network_io_performed"] is False
    payload = json.loads(queue.read_text(encoding="utf-8"))
    assert payload["schema"] == queue_processor.QUEUE_SCHEMA
    assert payload["submission_count_enqueued"] == 1
    assert len(payload["queue"]) == 1
    assert payload["queue"][0]["queue_status"] == "pending_explicit_direct_policy_uploader"

    again = queue_processor.enqueue_authorized_handoff(
        authorization_path=paths["authorization"],
        finalizer_receipt_path=paths["receipt"],
        queue_path=queue,
        contract_path=paths["contract"],
        official_libcg_staging_path=paths["staging"],
        enqueue=True,
    )
    assert again["status"] == "already_enqueued_idempotent"
    with pytest.raises(queue_processor.R241QueueProcessorError, match="already consumed"):
        queue_processor.enqueue_authorized_handoff(
            authorization_path=paths["authorization"],
            finalizer_receipt_path=paths["receipt"],
            queue_path=tmp_path / "second-queue.json",
            contract_path=paths["contract"],
            official_libcg_staging_path=paths["staging"],
            enqueue=True,
        )


def test_r241_queue_processor_has_no_upload_client() -> None:
    source = (
        ROOT / "scripts/process_alakazam_new_list_direct_r241_submission_queue.py"
    ).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "import requests" not in source
    assert "kaggle.api" not in source


def test_r241_direct_uploader_revalidates_the_exact_terminal_queue_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, paths = _finalize_fixture(tmp_path, monkeypatch)
    queue = tmp_path / "direct-policy-queue.json"
    queue_processor.enqueue_authorized_handoff(
        authorization_path=paths["authorization"],
        finalizer_receipt_path=paths["receipt"],
        queue_path=queue,
        contract_path=paths["contract"],
        official_libcg_staging_path=paths["staging"],
        enqueue=True,
    )

    result = uploader.process_once(
        queue_path=queue,
        authorization_path=paths["authorization"],
        finalizer_receipt_path=paths["receipt"],
        receipts_dir=tmp_path / "upload-receipts",
        kaggle=Path("/not-run-in-local-preflight/kaggle"),
        contract_path=paths["contract"],
        official_libcg_staging_path=paths["staging"],
        upload=False,
        allow_noncanonical_contract_for_test=True,
    )

    assert result["status"] == "local_preflight_passed_upload_not_requested"
    assert result["competition"] == "pokemon-tcg-ai-battle"
    assert result["turn_order_preference"] == "first_if_allowed"
    assert "FIRST IF ALLOWED" in result["submission_label"]
    assert not (tmp_path / "upload-receipts").exists()
