"""Focused, file-backed coverage for the r198 three-arm evaluator contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

import pytest

import poke_bot.rtp_three_arm_evaluation as evaluator
from poke_bot.engine_rebuild.rtp_pairing_snapshot import (
    PairingArtifactSet,
    emit_true_rng_pairing_capability,
    emit_true_rng_pairing_probe,
    frozen_file_identity,
    snapshot_abi_sha256,
)
from poke_bot.rtp_r198_evaluation_input_materializer import (
    CapturedSnapshot,
    _r197_canonical_json_digest,
    materialize_r198_evaluation_inputs,
)
from poke_bot.rtp_r198_production_factory import r198_runtime_profile_payload
from poke_bot.rtp_three_arm_evaluation import (
    RTPThreeArmEvaluationError,
    compile_three_arm_receipt,
    prepare_three_arm_manifest_from_spec,
)


ROOT = Path(__file__).resolve().parents[1]

# The canonical production pairing closure deliberately uses an empty package
# marker.  Keep this literal production identity in the fixture so the
# evaluator cannot regress by treating a regular file as necessarily nonempty.
ACTUAL_ZERO_BYTE_CG_INIT = {
    "relative_path": "__init__.py",
    "sha256": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "bytes": 0,
}


def _sha(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: bytes | Mapping[str, Any], *, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Mapping):
        path.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    path.chmod(mode)
    return path


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def _profile(arm: str) -> dict[str, Any]:
    return {
        "evaluation_arm": arm,
        "sizing_profile": "pure_rl_r197",
        "recursive_turn_planner_enabled": arm != "no_rtp",
        "direct_bridge_enabled": arm != "no_rtp",
        "force_direct_bridge_only": arm == evaluator.DIRECT_BRIDGE_ARM,
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_plan_length": 12,
        "d_model": 96,
        "dynamics_width": 192,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "repair_budget": 1,
    }


def _make_capability(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Generate v2 capability/probe bytes through the real emitter API."""

    engine = _write(tmp_path / "pairing-engine.so", b"test private pairing engine", mode=0o555)
    source = _write(tmp_path / "source-manifest.json", {"schema": "source", "files": []})
    patch = _write(tmp_path / "RtpPairingSnapshotExport.cpp", b"test private patch")
    build = _write(
        tmp_path / "pairing-build.json",
        {
            "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_build/v1",
            "status": "success",
            "engine_artifact_sha256": frozen_file_identity(engine)["sha256"],
            "source_artifact_sha256": frozen_file_identity(source)["sha256"],
            "patch_artifact_sha256": frozen_file_identity(patch)["sha256"],
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "engine_artifact": frozen_file_identity(engine),
            "source_artifact": frozen_file_identity(source),
            "patch_artifact": frozen_file_identity(patch),
        },
    )
    artifacts = PairingArtifactSet.from_paths(
        engine_path=engine,
        source_manifest_path=source,
        patch_path=patch,
        build_receipt_path=build,
    )
    probe = emit_true_rng_pairing_probe(
        output_path=tmp_path / "pairing-probe.json",
        artifacts=artifacts,
        deterministic_probe={
            "passed": True,
            "initial_snapshot_fingerprint_sha256": "sha256:" + "1" * 64,
            "initial_snapshot_fingerprint_bytes": 32,
            "deterministic_transcript_sha256": "sha256:" + "2" * 64,
            "transcript_steps": 2,
            "duplicate_restore_independent_handles": True,
            "device_rand_false_verified": True,
            "requested_seed_only_rejected": True,
            "delayed_restore_transcript_passed": True,
            "cross_process_restore_passed": True,
        },
        divergent_policy_true_pairing_passed=True,
        all_arms_restored_or_replayed=True,
    )
    capability = emit_true_rng_pairing_capability(
        output_path=tmp_path / "pairing-capability.json",
        artifacts=artifacts,
        probe_path=probe,
    )
    return capability, engine, build


def _make_evaluation_cg_closure(tmp_path: Path, *, engine: Path, build: Path) -> Path:
    engine_identity = _identity(engine)
    build_identity = _identity(build)
    paths = ("__init__.py", "api.py", "game.py", "libcg.so", "sim.py", "utils.py")

    def tree(path: Path, schema: str) -> Path:
        files = []
        for name in paths:
            if name == "__init__.py":
                files.append(dict(ACTUAL_ZERO_BYTE_CG_INIT))
            elif name == "libcg.so":
                files.append(
                    {
                        "relative_path": name,
                        "sha256": engine_identity["sha256"],
                        "bytes": engine_identity["bytes"],
                    }
                )
            else:
                files.append(
                    {
                        "relative_path": name,
                        "sha256": _sha(f"{schema}:{name}"),
                        "bytes": 1,
                    }
                )
        material = {"schema": schema, "file_count": len(files), "files": files}
        return _write(path, {**material, "tree_sha256": evaluator.canonical_digest(material)})

    source_manifest = tree(
        tmp_path / "cg-source-manifest.json",
        "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1",
    )
    closure_manifest = tree(
        tmp_path / "cg-closure-manifest.json",
        "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1",
    )
    public_engine = _write(tmp_path / "public-cg.so", b"public cg")
    raw_cards = _sha("raw cards")
    raw_attacks = _sha("raw attacks")
    parity = _write(
        tmp_path / "cg-metadata-parity.json",
        {
            "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1",
            "status": "passed",
            "independent_processes": True,
            "public_initialized_before_pairing": True,
            "pairing_private_initialize_after_public_passed": True,
            "distinct_dso_handles": True,
            "public_cg_engine": _identity(public_engine),
            "pairing_engine": engine_identity,
            "all_card_canonical_sha256": _sha("canonical cards"),
            "all_attack_canonical_sha256": _sha("canonical attacks"),
            "public_all_card_raw_sha256": raw_cards,
            "pairing_all_card_raw_sha256": raw_cards,
            "public_all_attack_raw_sha256": raw_attacks,
            "pairing_all_attack_raw_sha256": raw_attacks,
        },
    )
    return _write(
        tmp_path / "evaluation-cg-closure.json",
        {
            "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1",
            "status": "sealed",
            "engine_artifact": engine_identity,
            "pairing_build_artifact": build_identity,
            "cg_source_manifest": _identity(source_manifest),
            "closure_manifest": _identity(closure_manifest),
            "metadata_parity": _identity(parity),
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "sim_initializer_symbol": "RtpPairingSnapshotInitialize",
            "snapshot_abi_version": 2,
        },
    )


def _make_snapshot_runtime_library(tmp_path: Path, *, engine: Path) -> Path:
    """Make the physical 0444 DSO that evaluation children actually load.

    The closure's private pairing build artifact is intentionally executable
    and lives elsewhere; the production snapshot instead loads this sealed
    byte-identical copy via ``CG_LIB_PATH``.
    """

    return _write(
        tmp_path / "snapshot-eval-cg" / "cg" / "libcg.so",
        engine.read_bytes(),
        mode=0o444,
    )


@pytest.mark.unit
def test_evaluation_cg_closure_accepts_production_zero_byte_package_marker(
    tmp_path: Path,
) -> None:
    """Both fixed CG tree schemas permit the real empty ``__init__.py``."""

    capability_path, engine, build = _make_capability(tmp_path / "pairing")
    closure_path = _make_evaluation_cg_closure(
        tmp_path / "closure", engine=engine, build=build
    )
    runtime_library = _make_snapshot_runtime_library(tmp_path, engine=engine)
    pairing_capability = evaluator._normalize_pairing_capability(
        {"receipt": _identity(capability_path)}
    )

    closure_payload = json.loads(closure_path.read_text(encoding="utf-8"))
    for field in ("cg_source_manifest", "closure_manifest"):
        tree = json.loads(Path(str(closure_payload[field]["path"])).read_text(encoding="utf-8"))
        assert tree["files"][0] == ACTUAL_ZERO_BYTE_CG_INIT

    normalized = evaluator._normalize_evaluation_cg_closure(
        {
            "receipt": _identity(closure_path),
            "runtime_library": _identity(runtime_library),
        },
        pairing_capability,
    )

    assert normalized["cg_source_manifest"] == closure_payload["cg_source_manifest"]
    assert normalized["closure_manifest"] == closure_payload["closure_manifest"]
    assert normalized["runtime_library"] == _identity(runtime_library)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tree_field", "mutation", "expected_suffix"),
    (
        (
            "cg_source_manifest",
            "negative",
            "must be at least 0",
        ),
        (
            "cg_source_manifest",
            "boolean",
            "must be an integer",
        ),
        (
            "cg_source_manifest",
            "missing",
            "must be an integer",
        ),
        (
            "closure_manifest",
            "negative",
            "must be at least 0",
        ),
        (
            "closure_manifest",
            "boolean",
            "must be an integer",
        ),
        (
            "closure_manifest",
            "missing",
            "must be an integer",
        ),
    ),
)
def test_evaluation_cg_closure_rejects_invalid_zero_byte_marker_counts(
    tmp_path: Path,
    tree_field: str,
    mutation: str,
    expected_suffix: str,
) -> None:
    """Zero is valid, but the sealed tree never accepts missing/negative sizes."""

    capability_path, engine, build = _make_capability(tmp_path / "pairing")
    closure_path = _make_evaluation_cg_closure(
        tmp_path / "closure", engine=engine, build=build
    )
    runtime_library = _make_snapshot_runtime_library(tmp_path, engine=engine)
    pairing_capability = evaluator._normalize_pairing_capability(
        {"receipt": _identity(capability_path)}
    )

    closure_payload = json.loads(closure_path.read_text(encoding="utf-8"))
    tree_path = Path(str(closure_payload[tree_field]["path"]))
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    assert tree["files"][0] == ACTUAL_ZERO_BYTE_CG_INIT
    if mutation == "negative":
        tree["files"][0]["bytes"] = -1
    elif mutation == "boolean":
        tree["files"][0]["bytes"] = True
    else:
        assert mutation == "missing"
        tree["files"][0].pop("bytes")
    invalid_tree = _write(tmp_path / f"invalid-{tree_field}.json", tree)
    closure_payload[tree_field] = _identity(invalid_tree)
    invalid_closure = _write(tmp_path / f"invalid-{tree_field}-closure.json", closure_payload)
    expected_message = (
        f"evaluation CG closure {tree_field}.files[0].bytes {expected_suffix}"
    )

    with pytest.raises(RTPThreeArmEvaluationError, match=re.escape(expected_message)):
        evaluator._normalize_evaluation_cg_closure(
            {
                "receipt": _identity(invalid_closure),
                "runtime_library": _identity(runtime_library),
            },
            pairing_capability,
        )


def _make_completion(tmp_path: Path, candidate_contract_sha256: str) -> Path:
    train = ["r197-train-a", "r197-train-b"]
    heldout = ["r197-heldout-a"]

    def side(split: str, ids: list[str]) -> dict[str, Any]:
        return {
            "candidate_selection": {
                "schema": "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1",
                "split": split,
            },
            "batch_cap_selection": {
                "retained_episode_count": len(ids),
                "retained_episode_ids_sha256": _r197_canonical_json_digest(ids),
                "row_level_sampling": False,
                "cross_window_dynamics_target": False,
            },
            "retained_episode_ids": ids,
        }

    selection = {
        "schema": "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1",
        "row_level_sampling": False,
        "cross_window_dynamics_target": False,
        "selection_plan_sha256": _sha("selection plan"),
        "train_selection_sha256": _r197_canonical_json_digest(train),
        "heldout_selection_sha256": _r197_canonical_json_digest(heldout),
        "train": side("train", train),
        "heldout": side("heldout", heldout),
    }
    return _write(
        tmp_path / "r197-completion.json",
        {
            "schema": "poke_bot.alakazam_rtp_r197_shadow_candidate/v1",
            "status": "completed_shadow_only",
            "candidate_contract_sha256": candidate_contract_sha256,
            "authority": {
                "shadow_only": True,
                "serving_eligible": False,
                "action_authority_enabled": False,
                "selector_authority": False,
                "live_checkpoint_publication": False,
                "submission_eligible": False,
            },
            "contract": {
                "complete_action_corpus": {
                    "schema": "poke_bot.rtp_complete_action_shadow_corpus/v1",
                    "manifest_sha256": _sha("r197 corpus manifest"),
                    "receipt_sha256": _sha("r197 corpus receipt"),
                    "split": {"source_disjoint": True, "unit": "episode_id", "seed": 5_000_000},
                    "selection": selection,
                }
            },
            "training": {
                "heldout_is_source_excluded": True,
                "candidate_target_wiring": {
                    "status": "masked_absent_no_fabrication",
                    "latent_lookahead_targets": "not_wired_future_input",
                    "unobserved_action_returns": "not_fabricated",
                    "value_of_planning_target": "not_heuristic_labeled",
                },
                "metrics": {
                    "rtp_heldout": {
                        "mean_candidate_calibration_target_count": 0.0,
                        "mean_candidate_ranking_pair_count": 0.0,
                        "mean_candidate_return_target_count": 0.0,
                    }
                },
            },
        },
    )


def _sealed_opponent(tmp_path: Path, opponent_id: str, content_digest: str, offset: int) -> dict[str, Any]:
    root = tmp_path / "sealed-opponents" / opponent_id
    root.mkdir(parents=True)
    deck = root / "deck.csv"
    deck.write_text("\n".join(str(offset + index) for index in range(60)) + "\n", encoding="utf-8")
    main = root / "main.py"
    main.write_text("def select(*_args):\n    return [0]\n", encoding="utf-8")
    deck.chmod(0o444)
    main.chmod(0o444)
    root.chmod(0o555)
    entries = [
        {"path": child.name, "sha256": _sha(child.read_bytes()), "bytes": child.stat().st_size}
        for child in sorted((deck, main), key=lambda item: item.name)
    ]
    artifact = _write(
        tmp_path / "opponent-manifests" / f"{opponent_id}.json",
        {
            "schema": "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1",
            "status": "sealed",
            "opponent_id": opponent_id,
            "content_digest": content_digest,
            "package_root": str(root),
            "no_symlinks": True,
            "all_paths_read_only": True,
            "deck_sha256": _sha(deck.read_bytes()),
            "deck_order_sha256": _sha(deck.read_bytes()),
            "entries": entries,
            "tree_entries_sha256": evaluator.canonical_digest(entries),
        },
    )
    return {
        "id": opponent_id,
        "content_digest": content_digest,
        "deck": _identity(deck),
        "artifact": _identity(artifact),
        "package_root": str(root),
    }


def _patch_test_candidate_constants(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_contract: str,
    parent: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    deck: Mapping[str, Any],
    matchup_tree: Mapping[str, Any],
) -> dict[str, str]:
    values = {
        "R198_CANDIDATE_CONTRACT_SHA256": candidate_contract,
        "R198_PARENT_CHECKPOINT_SHA256": str(parent["sha256"]),
        "R198_SIDECAR_SHA256": str(sidecar["sha256"]),
        "R198_SIDECAR_CONFIG_SHA256": _sha("test sidecar config"),
        "R198_DECK_FILE_SHA256": str(deck["sha256"]),
        "R198_DECK_CARDS_SHA256": _sha("test canonical deck cards"),
        "R198_MATCHUP_TREE_SHA256": str(matchup_tree["sha256"]),
    }
    import poke_bot.rtp_r198_evaluation_input_materializer as materializer

    for name, value in values.items():
        monkeypatch.setattr(evaluator, name, value)
        monkeypatch.setattr(materializer, name, value)
    return values


def _materialized_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Materialize the actual 1,000 sealed snapshot inputs then prepare v2."""

    tmp_path = Path(os.path.realpath(tmp_path))
    parent = _identity(_write(tmp_path / "candidate" / "parent.pt", b"parent checkpoint"))
    deck = _identity(
        _write(
            tmp_path / "candidate" / "deck.csv",
            ("\n".join(str(1000 + index) for index in range(60)) + "\n").encode(),
        )
    )
    matchup_tree = _identity(_write(tmp_path / "candidate" / "matchup-tree.json", {"tree": "frozen"}))
    sidecar = _identity(_write(tmp_path / "candidate" / "rtp-sidecar.pt", b"rtp sidecar"))
    candidate_contract = _sha("test r198 candidate contract")
    candidate = _patch_test_candidate_constants(
        monkeypatch,
        candidate_contract=candidate_contract,
        parent=parent,
        sidecar=sidecar,
        deck=deck,
        matchup_tree=matchup_tree,
    )
    capability, engine, build = _make_capability(tmp_path / "pairing")
    closure = _make_evaluation_cg_closure(tmp_path / "closure", engine=engine, build=build)
    runtime_library = _make_snapshot_runtime_library(tmp_path, engine=engine)
    matchup_adapter_registry = _identity(
        _write(
            tmp_path / "state" / "matchup_adapter_roster.json",
            (ROOT / "state" / "matchup_adapter_roster.json").read_bytes(),
        )
    )
    completion = _make_completion(tmp_path / "candidate", candidate_contract)
    registry = _write(
        tmp_path / "registry.json",
        (ROOT / "ops" / "research_control_registry_v1.json").read_bytes(),
    )
    opponents = [
        _sealed_opponent(tmp_path, opponent_id, digest, 2_000 + index * 100)
        for index, (opponent_id, digest) in enumerate(
            sorted(evaluator.R198_OFFICIAL_CONTROL_OPPONENTS.items())
        )
    ]
    arms: dict[str, dict[str, Any]] = {}
    for arm in evaluator.ARMS:
        runtime = _identity(_write(tmp_path / "runtime" / f"{arm}.bin", arm.encode()))
        arms[arm] = {
            "runtime_artifact": runtime,
            "runtime_profile_payload": r198_runtime_profile_payload(arm),
        }
        if arm != "no_rtp":
            arms[arm]["rtp_sidecar"] = sidecar
    binding = {
        "schema": evaluator.CANDIDATE_EVALUATION_BINDING_SCHEMA,
        "status": "bound",
        "candidate_contract_sha256": candidate["R198_CANDIDATE_CONTRACT_SHA256"],
        "parent_checkpoint_sha256": candidate["R198_PARENT_CHECKPOINT_SHA256"],
        "sidecar_sha256": candidate["R198_SIDECAR_SHA256"],
        "sidecar_config_sha256": candidate["R198_SIDECAR_CONFIG_SHA256"],
        "deck_file_sha256": candidate["R198_DECK_FILE_SHA256"],
        "deck_cards_sha256": candidate["R198_DECK_CARDS_SHA256"],
        "matchup_tree_sha256": candidate["R198_MATCHUP_TREE_SHA256"],
        "sizing_profile": "pure_rl_r197",
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "required_neural_passes": {"normal": 6, "forced_replan": 5},
    }
    base_spec = _write(
        tmp_path / "base-spec.json",
        {
            "shared_artifacts": {
                "parent_checkpoint": parent,
                "deck": deck,
                "matchup_tree": matchup_tree,
            },
            "arms": arms,
            "opponents": opponents,
            "production_factory": {
                "source_snapshot_root": str(tmp_path),
                "artifacts": {"deck": deck},
                "evaluation_cg": {"library": _identity(runtime_library)},
                "matchup_adapter_registry": {
                    **matchup_adapter_registry,
                    "mode": 0o444,
                },
            },
            "candidate_evaluation_binding": binding,
            "evaluation_cg_closure": {
                "receipt": _identity(closure),
                "runtime_library": _identity(runtime_library),
            },
        },
    )
    captured = 0

    def capture(_deck0: object, _deck1: object, _seed: int) -> CapturedSnapshot:
        nonlocal captured
        payload = f"opaque-native-snapshot-{captured}".encode("utf-8")
        captured += 1
        return CapturedSnapshot(
            serialized_bytes=payload,
            snapshot_id=f"snapshot-{_sha(payload)[7:31]}",
            fingerprint_sha256=_sha(payload),
            fingerprint_bytes=len(payload),
        )

    def preflight(preflight_input: Mapping[str, Any], output_path: Path) -> Path:
        direct = preflight_input["arms"][evaluator.DIRECT_BRIDGE_ARM]
        recursive = preflight_input["arms"]["recursive_rtp"]
        return _write(
            output_path,
            {
                "schema": "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1",
                "status": "passed",
                "sidecar_sha256": direct["rtp_sidecar"]["sha256"],
                "direct_runtime_profile_sha256": direct["runtime_profile"]["sha256"],
                "recursive_runtime_profile_sha256": recursive["runtime_profile"]["sha256"],
                "max_neural_passes": 256,
                "max_action_combos": 1024,
                "normal_probe_observed_neural_passes": 6,
                "forced_replan_probe_observed_neural_passes": 5,
                "normal_probe_completed": True,
                "forced_replan_probe_completed": True,
                "neural_budget_failures": 0,
                "matchup_adapter_registry_sha256": (
                    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
                ),
                "matchup_adapter_slot_registry_digest": (
                    "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
                ),
            },
        )

    seeds = iter(range(700_000, 702_000))
    materialized = materialize_r198_evaluation_inputs(
        completion_receipt=completion,
        research_control_registry=registry,
        pairing_capability=capability,
        evaluator_base_spec=base_spec,
        output_root=tmp_path / "evaluation-inputs",
        run_nonce="three-arm-evaluator-unit",
        snapshot_capturer=capture,
        preflight_runner=preflight,
        seed_provider=lambda: next(seeds),
        fixture_observation_extractor=lambda _seal: {"current": {"result": -1}},
    )
    assert captured == 1_002
    manifest_path = Path(str(materialized["prepared_evaluator_manifest"]["path"]))
    return {
        "materialized": materialized,
        "manifest_path": manifest_path,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "candidate": candidate,
    }


def _forced_turn_order_trace() -> list[dict[str, Any]]:
    """The frozen snapshot asks physical player 0 to take Yes at index zero."""

    return [
        {
            "control_index": 0,
            "control": "forced_go_first_contract",
            "prompt_context": 41,
            "prompt_context_encoding": "numeric_41",
            "expected_action": [0],
            "returned_action": [0],
            "verified_observation_action_contract": True,
            "rtp_diagnostics_absent": True,
            "complexity_probe_not_invoked": True,
            "excluded_from_candidate_decisions": True,
            "excluded_from_intended_complex_denominator": True,
            "excluded_from_latency": True,
        }
    ]


def _telemetry(arm: str, *, candidate_seat: int) -> dict[str, Any]:
    if candidate_seat not in {0, 1}:
        raise AssertionError("fixture has an invalid candidate seat")
    successful = arm == "recursive_rtp"
    if arm == "no_rtp":
        planner_mode, mode, reason, latency = "no_rtp", "no_rtp", "baseline", 0.010
    elif arm == evaluator.DIRECT_BRIDGE_ARM:
        planner_mode, mode, reason, latency = "direct_bridge", evaluator.DIRECT_BRIDGE_ARM, "bridge", 0.020
    else:
        planner_mode, mode, reason, latency = "recursive_plan", "recursive_rtp", "complex", 0.030
    return {
        "candidate_decisions": 1,
        "planner_eligible_candidate_decisions": 1,
        "over_cap_factorized_fallback_decisions": 0,
        "over_cap_factorized_fallback_trace": [],
        "over_cap_factorized_fallback_trace_sha256": evaluator.canonical_digest([]),
        "forced_turn_order_controls": int(candidate_seat == 0),
        "forced_turn_order_control_trace": (
            _forced_turn_order_trace() if candidate_seat == 0 else []
        ),
        "intended_complex_decisions": int(successful),
        "recursive_intended_complex_decisions": int(successful),
        "successful_recursive_intended_complex_decisions": int(successful),
        "direct_bridge_decisions": int(arm == evaluator.DIRECT_BRIDGE_ARM),
        "recursive_decisions": int(successful),
        "fallback_decisions": 0,
        "unexpected_recursive_fallback_decisions": 0,
        "expected_recursive_fallback_decisions": 0,
        "neural_budget_exceeded": 0,
        "neural_budget_failures": 0,
        "illegal_action_count": 0,
        "candidate_forfeit_count": 0,
        "intended_complex_decision_scope": "new_turn_complexity_gate_only",
        "recursive_mode_counts": {
            "continue_plan": 0,
            "direct_policy_fallback": 0,
            "recursive_plan": int(successful),
            "replan_direct": 0,
            "replan_with_program": 0,
        },
        "latency_seconds": latency,
        "decision_latency_trace": [
            {
                "decision_index": 0,
                "mode": mode,
                "planner_mode": planner_mode,
                "planner_reason": reason,
                "intended_complex": successful,
                "fallback_classification": None,
                "latency_seconds": latency,
            }
        ],
        "normal_recursive_plan_passes": [6] if successful else [],
        "forced_replan_passes": [],
    }


def _over_cap_telemetry(arm: str, *, candidate_seat: int) -> dict[str, Any]:
    """One exact non-materializing n=9/counts=1..5 telemetry row."""

    telemetry = _telemetry(arm, candidate_seat=candidate_seat)
    action_space = {
        "n_options": 9,
        "min_count": 1,
        "max_count": 5,
        "counts": [1, 2, 3, 4, 5],
        "complete_ordered_action_cardinality": 18_729,
        "complete_ordered_action_cap": 1024,
        "over_cap": True,
        "complete_ordered_actions_materialized": False,
        "complete_ordered_action_truncated": False,
    }
    action_space_sha256 = evaluator.canonical_digest(action_space)
    observation_sha256 = "sha256:" + "b" * 64
    policy_input_sha256 = "sha256:" + "c" * 64
    logical_pre_action_sha256 = evaluator.canonical_digest(
        {
            "observation_sha256": observation_sha256,
            "action_space_sha256": action_space_sha256,
            "candidate_policy_input_sha256": policy_input_sha256,
        }
    )
    diagnostic: dict[str, Any] | None
    if arm == "no_rtp":
        diagnostic = None
    else:
        diagnostic = {
            "mode": "fallback",
            "fallback_code": "action_space_too_large",
            "neural_passes": 0,
            "required_neural_passes": 0,
            "legal_count": 0,
            "decision_mode": "",
        }
    trace = {
        "decision_index": 0,
        "arm": arm,
        "mode": "over_cap_factorized_fallback",
        "classification": "complete_ordered_action_space_over_cap",
        "action_space": action_space,
        "action_space_sha256": action_space_sha256,
        "observation_sha256": observation_sha256,
        "candidate_policy_input_sha256": policy_input_sha256,
        "logical_pre_action_sha256": logical_pre_action_sha256,
        "returned_action": [0, 2, 3, 6, 5],
        "factorized_teacher_forcing_legal": True,
        "factorized_teacher_forcing_stage_count": 5,
        "complexity_probe_not_invoked": True,
        "neural_passes": 0,
        "required_neural_passes": 0,
        "neural_budget_failure": False,
        "rtp_diagnostic": diagnostic,
        "included_in_candidate_decisions": True,
        "included_in_candidate_latency": True,
        "excluded_from_planner_eligible_candidate_decisions": True,
        "excluded_from_intended_complex_denominator": True,
        "excluded_from_direct_bridge_metrics": True,
        "excluded_from_recursive_metrics": True,
        "excluded_from_fallback_metrics": True,
        "excluded_from_neural_pass_metrics": True,
        "excluded_from_recursive_latency": True,
    }
    telemetry.update(
        {
            "planner_eligible_candidate_decisions": 0,
            "over_cap_factorized_fallback_decisions": 1,
            "over_cap_factorized_fallback_trace": [trace],
            "over_cap_factorized_fallback_trace_sha256": evaluator.canonical_digest([trace]),
            "intended_complex_decisions": 0,
            "recursive_intended_complex_decisions": 0,
            "successful_recursive_intended_complex_decisions": 0,
            "direct_bridge_decisions": 0,
            "recursive_decisions": 0,
            "fallback_decisions": 0,
            "unexpected_recursive_fallback_decisions": 0,
            "expected_recursive_fallback_decisions": 0,
            "normal_recursive_plan_passes": [],
            "forced_replan_passes": [],
            "recursive_mode_counts": {
                "continue_plan": 0,
                "direct_policy_fallback": 0,
                "recursive_plan": 0,
                "replan_direct": 0,
                "replan_with_program": 0,
            },
            "latency_seconds": 0.040,
            "decision_latency_trace": [
                {
                    "decision_index": 0,
                    "mode": "over_cap_factorized_fallback",
                    "planner_mode": "over_cap_factorized_fallback",
                    "planner_reason": "complete_ordered_action_space_over_cap",
                    "intended_complex": None,
                    "fallback_classification": None,
                    "latency_seconds": 0.040,
                    "over_cap_trace_index": 0,
                }
            ],
        }
    )
    return telemetry


@pytest.mark.unit
@pytest.mark.parametrize("arm", evaluator.ARMS)
def test_over_cap_factorized_telemetry_is_strict_and_planner_ineligible(arm: str) -> None:
    telemetry = evaluator._result_telemetry(
        _over_cap_telemetry(arm, candidate_seat=1), arm=arm, label="over cap"
    )
    assert telemetry["candidate_decisions"] == 1
    assert telemetry["planner_eligible_candidate_decisions"] == 0
    assert telemetry["over_cap_factorized_fallback_decisions"] == 1
    assert telemetry["fallback_decisions"] == 0
    assert telemetry["intended_complex_decisions"] == 0
    assert telemetry["recursive_decision_latency_seconds"] == []

    bad_cardinality = _over_cap_telemetry(arm, candidate_seat=1)
    bad_cardinality["over_cap_factorized_fallback_trace"][0]["action_space"][
        "complete_ordered_action_cardinality"
    ] = 1025
    bad_cardinality["over_cap_factorized_fallback_trace_sha256"] = evaluator.canonical_digest(
        bad_cardinality["over_cap_factorized_fallback_trace"]
    )
    with pytest.raises(RTPThreeArmEvaluationError, match="cardinality does not recompute"):
        evaluator._result_telemetry(bad_cardinality, arm=arm, label="bad cardinality")

    coerced_bound = _over_cap_telemetry(arm, candidate_seat=1)
    coerced_bound["over_cap_factorized_fallback_trace"][0]["action_space"]["min_count"] = 1.0
    coerced_bound["over_cap_factorized_fallback_trace_sha256"] = evaluator.canonical_digest(
        coerced_bound["over_cap_factorized_fallback_trace"]
    )
    with pytest.raises(RTPThreeArmEvaluationError, match="exact non-bool integer"):
        evaluator._result_telemetry(coerced_bound, arm=arm, label="coerced bound")

    assessed_false = _over_cap_telemetry(arm, candidate_seat=1)
    assessed_false["decision_latency_trace"][0]["intended_complex"] = False
    with pytest.raises(RTPThreeArmEvaluationError, match="entered planner/fallback accounting"):
        evaluator._result_telemetry(assessed_false, arm=arm, label="assessed false")


@pytest.mark.unit
def test_over_cap_cross_arm_action_parity_is_conditional_on_logical_input() -> None:
    def paired_rows() -> dict[str, dict[str, dict[str, Any]]]:
        return {
            "cell-000000": {
                arm: {
                    "telemetry": _over_cap_telemetry(arm, candidate_seat=1),
                }
                for arm in evaluator.ARMS
            }
        }

    same = paired_rows()
    observed = evaluator._validate_conditional_over_cap_action_parity(same)
    assert observed == {
        "over_cap_trace_rows": 3,
        "logical_pre_action_groups": 1,
        "cross_arm_comparable_groups": 1,
        "cross_arm_comparable_arm_rows": 3,
    }

    divergent = paired_rows()
    for index, arm in enumerate(evaluator.ARMS):
        trace = divergent["cell-000000"][arm]["telemetry"]["over_cap_factorized_fallback_trace"][0]
        trace["candidate_policy_input_sha256"] = "sha256:" + str(index + 1) * 64
        trace["logical_pre_action_sha256"] = evaluator.canonical_digest(
            {
                "observation_sha256": trace["observation_sha256"],
                "action_space_sha256": trace["action_space_sha256"],
                "candidate_policy_input_sha256": trace["candidate_policy_input_sha256"],
            }
        )
        trace["returned_action"] = [index]
    assert evaluator._validate_conditional_over_cap_action_parity(divergent)[
        "cross_arm_comparable_groups"
    ] == 0

    mismatch = paired_rows()
    mismatch["cell-000000"][evaluator.DIRECT_BRIDGE_ARM]["telemetry"][
        "over_cap_factorized_fallback_trace"
    ][0]["returned_action"] = [1, 2, 3, 4, 5]
    with pytest.raises(RTPThreeArmEvaluationError, match="actions differ"):
        evaluator._validate_conditional_over_cap_action_parity(mismatch)


@pytest.mark.unit
def test_forced_turn_order_trace_is_exact_seat_bound_and_tamper_evident() -> None:
    raw = _telemetry("no_rtp", candidate_seat=0)
    telemetry = evaluator._result_telemetry(raw, arm="no_rtp", label="forced control")
    evaluator._validate_r198_forced_turn_order_contract(
        telemetry, candidate_seat=0, label="forced control"
    )
    assert telemetry["candidate_decisions"] == 1
    assert telemetry["forced_turn_order_controls"] == 1
    assert telemetry["forced_turn_order_control_trace"] == _forced_turn_order_trace()

    wrong_action = json.loads(json.dumps(raw))
    wrong_action["forced_turn_order_control_trace"][0]["returned_action"] = [1]
    with pytest.raises(RTPThreeArmEvaluationError, match="does not equal"):
        evaluator._result_telemetry(wrong_action, arm="no_rtp", label="wrong action")

    bool_action = json.loads(json.dumps(raw))
    bool_action["forced_turn_order_control_trace"][0]["expected_action"] = [True]
    with pytest.raises(RTPThreeArmEvaluationError, match="action index"):
        evaluator._result_telemetry(bool_action, arm="no_rtp", label="bool action")

    float_context = json.loads(json.dumps(raw))
    float_context["forced_turn_order_control_trace"][0]["prompt_context"] = 41.0
    with pytest.raises(RTPThreeArmEvaluationError, match="invalid prompt context"):
        evaluator._result_telemetry(float_context, arm="no_rtp", label="float context")

    float_control_index = json.loads(json.dumps(raw))
    float_control_index["forced_turn_order_control_trace"][0]["control_index"] = 0.0
    with pytest.raises(RTPThreeArmEvaluationError, match="control indexes"):
        evaluator._result_telemetry(
            float_control_index, arm="no_rtp", label="float control index"
        )

    float_action = json.loads(json.dumps(raw))
    float_action["forced_turn_order_control_trace"][0]["expected_action"] = [0.0]
    with pytest.raises(RTPThreeArmEvaluationError, match="action index"):
        evaluator._result_telemetry(float_action, arm="no_rtp", label="float action")

    string_action = json.loads(json.dumps(raw))
    string_action["forced_turn_order_control_trace"][0]["returned_action"] = ["0"]
    with pytest.raises(RTPThreeArmEvaluationError, match="action index"):
        evaluator._result_telemetry(string_action, arm="no_rtp", label="string action")

    probe_tamper = json.loads(json.dumps(raw))
    probe_tamper["forced_turn_order_control_trace"][0][
        "complexity_probe_not_invoked"
    ] = False
    with pytest.raises(RTPThreeArmEvaluationError, match="complexity_probe_not_invoked"):
        evaluator._result_telemetry(probe_tamper, arm="no_rtp", label="probe tamper")

    planner_tamper = json.loads(json.dumps(raw))
    planner_tamper["forced_turn_order_control_trace"][0]["intended_complex"] = True
    with pytest.raises(RTPThreeArmEvaluationError, match="exact canonical field set"):
        evaluator._result_telemetry(planner_tamper, arm="no_rtp", label="planner tamper")

    latency_tamper = json.loads(json.dumps(raw))
    latency_tamper["latency_seconds"] = 0.011
    with pytest.raises(RTPThreeArmEvaluationError, match="candidate decision trace total"):
        evaluator._result_telemetry(latency_tamper, arm="no_rtp", label="latency tamper")

    no_control = evaluator._result_telemetry(
        _telemetry("no_rtp", candidate_seat=1), arm="no_rtp", label="seat one"
    )
    with pytest.raises(RTPThreeArmEvaluationError, match="frozen seat ABI"):
        evaluator._validate_r198_forced_turn_order_contract(
            no_control, candidate_seat=0, label="missing seat zero control"
        )
    with pytest.raises(RTPThreeArmEvaluationError, match="frozen seat ABI"):
        evaluator._validate_r198_forced_turn_order_contract(
            telemetry, candidate_seat=1, label="unexpected seat one control"
        )

    paired = {
        "cell-000000": {
            arm: {"telemetry": json.loads(json.dumps(telemetry))}
            for arm in evaluator.ARMS
        }
    }
    evaluator._verify_forced_turn_order_pairing(paired)
    paired["cell-000000"][evaluator.DIRECT_BRIDGE_ARM]["telemetry"][
        "forced_turn_order_control_trace"
    ][0]["returned_action"] = [1]
    with pytest.raises(RTPThreeArmEvaluationError, match="trace differs across arms"):
        evaluator._verify_forced_turn_order_pairing(paired)


def _run_shaped_results(tmp_path: Path, manifest: Mapping[str, Any]) -> Path:
    evidence_root = tmp_path / "runner-shaped-evidence"
    rows: list[dict[str, Any]] = []
    opponents = {str(row["id"]): row for row in manifest["opponents"]}
    closure = manifest["evaluation_cg_closure"]
    for cell in manifest["schedule"]:
        cell_id = str(cell["cell_id"])
        candidate_rng = _sha(f"candidate-rng:{cell_id}")
        opponent_rng = _sha(f"opponent-rng:{cell_id}")
        common_environment = _sha(f"common-environment:{cell_id}")
        for arm in manifest["arm_order"]:
            arm_spec = manifest["arms"][arm]
            profile = arm_spec["profile"]
            sidecar = arm_spec["rtp_sidecar"]
            runtime = {
                "arm": arm,
                "runtime_artifact_sha256": arm_spec["runtime_artifact"]["sha256"],
                "runtime_profile_sha256": arm_spec["runtime_profile"]["sha256"],
                "action_attached_rtp_sidecar_sha256": None if sidecar is None else sidecar["sha256"],
                "complexity_probe_sidecar_sha256": manifest["arms"][evaluator.DIRECT_BRIDGE_ARM]["rtp_sidecar"]["sha256"],
                "complexity_probe_sidecar_instrumentation_only": True,
                "complexity_probe_latency_excluded": True,
                "rtp_action_attachment_enabled": arm != "no_rtp",
                "rtp_action_authority_enabled": False,
                **{
                    f"{name}_sha256": identity["sha256"]
                    for name, identity in manifest["shared_artifacts"].items()
                },
                "recursive_turn_planner_enabled": profile["recursive_turn_planner_enabled"],
                "direct_bridge_enabled": profile["direct_bridge_enabled"],
                "force_direct_bridge_only": profile["force_direct_bridge_only"],
                "max_neural_passes": profile["max_neural_passes"],
                "max_action_combos": profile["max_action_combos"],
            }
            telemetry = _telemetry(arm, candidate_seat=int(cell["candidate_seat"]))
            winner = "candidate" if arm == "recursive_rtp" else "opponent"
            terminal = {
                "winner": winner,
                "engine_result_code": int(cell["candidate_seat"]) if winner == "candidate" else 1 - int(cell["candidate_seat"]),
                "candidate_forfeit": False,
                "termination": "completed",
                "failed_seat": None,
                "engine_error": None,
                "candidate_error": None,
                "opponent_error": None,
            }
            score = 1.0 if winner == "candidate" else 0.0
            rng = {**cell["rng_identity"], "restored_or_replayed": True}
            opponent = opponents[str(cell["opponent_id"])]
            tag = f"{cell_id}-{arm}"
            transcript = _identity(_write(evidence_root / "transcripts" / f"{tag}.json", {"tag": tag}))
            action_context = None if arm == "no_rtp" else _sha(f"action-context:{tag}")
            isolation = {
                "launch_mode": "subprocess_exec",
                "fresh_process_per_arm": True,
                "process_model_load": True,
                "fresh_candidate_agent": True,
                "candidate_reset_called": True,
                "fresh_opponent_module": True,
                "engine_restore_before_first_select": True,
                "no_remote_leaf_sampling_mcts": True,
                "package_snapshot_verified_before_import": True,
                "complexity_probe_latency_excluded": True,
                "process_id": f"process:{tag}",
                "launch_nonce": f"nonce:{tag}",
                "baseline_content_digest": opponent["content_digest"],
                "baseline_package_root": opponent["package_root"],
                "baseline_tree_entries_sha256": opponent["tree_entries_sha256"],
                "baseline_package_manifest_sha256": opponent["artifact"]["sha256"],
                "baseline_deck_sha256": opponent["deck_sha256"],
                "candidate_rng_initial_state_sha256": candidate_rng,
                "opponent_rng_initial_state_sha256": opponent_rng,
                "opponent_rng_deterministic_or_no_rng": False,
                "common_sanitized_environment_sha256": common_environment,
                "arm_environment_sha256": _sha(f"arm-environment:{tag}"),
                "evaluation_cg_closure_receipt_sha256": closure["receipt"]["sha256"],
                "evaluation_cg_engine_sha256": closure["runtime_library"]["sha256"],
                "evaluation_cg_engine_path": closure["runtime_library"]["path"],
                "evaluation_cg_engine_bytes": closure["runtime_library"]["bytes"],
                "evaluation_cg_closure_manifest_sha256": closure["closure_manifest"]["sha256"],
                "evaluation_cg_metadata_parity_sha256": closure["metadata_parity"]["sha256"],
                "engine_loaded_path": closure["runtime_library"]["path"],
                "candidate_runtime_contract_sha256": _sha(f"candidate-runtime:{tag}"),
                "action_fence_sha256": (
                    None if arm == "no_rtp" else _sha(f"action-fence:{tag}")
                ),
                "evaluation_action_execution_sha256": action_context,
            }
            execution = {
                "schema": evaluator.EXECUTION_RECEIPT_SCHEMA,
                "status": "completed",
                "cell_id": cell_id,
                "arm": arm,
                "opponent_id": cell["opponent_id"],
                "candidate_seat": cell["candidate_seat"],
                "evaluation_case_id": cell["evaluation_case_id"],
                "evaluation_case_bindings_sha256": cell["evaluation_case_bindings_sha256"],
                "evaluation_corpus_sha256": manifest["r197_source_exclusion_binding"]["evaluation_only_cohort"]["sha256"],
                "transcript_sha256": transcript["sha256"],
                "runtime_identity_sha256": evaluator.canonical_digest(runtime),
                "rng_identity_sha256": evaluator.canonical_digest(rng),
                "telemetry_sha256": evaluator.canonical_digest(telemetry),
                "terminal_outcome_sha256": evaluator.canonical_digest(terminal),
                "candidate_score": score,
                "termination": "completed",
                "failed_seat": None,
                "engine_error": None,
                "candidate_error": None,
                "opponent_error": None,
                **isolation,
                "isolation": dict(isolation),
            }
            execution_identity = _identity(
                _write(evidence_root / "execution" / f"{tag}.json", execution)
            )
            rows.append(
                {
                    "cell_id": cell_id,
                    "arm": arm,
                    "opponent_id": cell["opponent_id"],
                    "candidate_seat": cell["candidate_seat"],
                    "evaluation_case_id": cell["evaluation_case_id"],
                    "evaluation_case_bindings_sha256": cell["evaluation_case_bindings_sha256"],
                    "evaluation_corpus_sha256": manifest["r197_source_exclusion_binding"]["evaluation_only_cohort"]["sha256"],
                    "completed": True,
                    "invalid": False,
                    "error": None,
                    "candidate_score": score,
                    "runtime_identity": runtime,
                    "rng_identity": rng,
                    "terminal_outcome": terminal,
                    "telemetry": telemetry,
                    "execution_receipt": execution_identity,
                    "transcript": transcript,
                }
            )
    return _write(evidence_root / "results.json", {"rows": rows})


@pytest.mark.unit
def test_r198_v2_prepares_runner_shaped_results_and_stays_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _materialized_fixture(tmp_path, monkeypatch)
    manifest = fixture["manifest"]
    manifest_path = fixture["manifest_path"]
    assert manifest["arm_order"] == list(evaluator.ARMS)
    assert len(manifest["schedule"]) == 1_000
    assert manifest["production_factory"]["evaluation_cg"]["library"] == manifest[
        "evaluation_cg_closure"
    ]["runtime_library"]
    assert manifest["production_factory"]["matchup_adapter_registry"] == {
        "path": str(tmp_path / "state" / "matchup_adapter_roster.json"),
        "sha256": "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc",
        "bytes": 11_899,
        "mode": 0o444,
    }
    assert manifest["planner_pass_preflight"]["matchup_adapter_registry_sha256"] == (
        "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
    )
    assert manifest["planner_pass_preflight"]["matchup_adapter_slot_registry_digest"] == (
        "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
    )
    assert manifest["evaluation_cg_closure"]["receipt"]["path"]
    assert (
        manifest["evaluation_cg_closure"]["runtime_library"]["path"]
        != manifest["evaluation_cg_closure"]["engine_artifact"]["path"]
    )
    assert (
        manifest["evaluation_cg_closure"]["runtime_library"]["sha256"]
        == manifest["evaluation_cg_closure"]["engine_artifact"]["sha256"]
    )
    assert stat.S_IMODE(
        Path(manifest["evaluation_cg_closure"]["runtime_library"]["path"]).stat().st_mode
    ) == 0o444
    assert manifest["r197_source_exclusion_binding"]["source_exclusion_computation"][
        "intersection_episode_count"
    ] == 0
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444

    results_path = _run_shaped_results(tmp_path, manifest)
    assert json.loads(results_path.read_text(encoding="utf-8"))["rows"][0][
        "terminal_outcome"
    ]["winner"] == "opponent"
    receipt_path = compile_three_arm_receipt(
        manifest_path=manifest_path,
        results=results_path,
        output_path=tmp_path / "compiled-receipt.json",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "hold"
    assert receipt["promotion_decision"]["self_promotion_performed"] is False
    assert receipt["promotion_gates"]["immutable_file_backed_execution_evidence"]["passed"] is True
    assert receipt["promotion_gates"]["recursive_path_exercised"]["passed"] is True
    assert receipt["promotion_gates"]["recursive_share_of_intended_complex_decisions"]["passed"] is True
    assert receipt["promotion_gates"]["unexpected_recursive_fallback_rate"]["passed"] is True
    assert receipt["promotion_gates"]["recursive_effect_lower_bound"]["passed"] is True
    assert receipt["promotion_gates"]["trusted_counterfactual_candidate_targets"]["passed"] is False
    assert receipt["promotion_gates"]["recursive_p95_decision_latency_slo"]["passed"] is False
    assert {
        arm: receipt["arm_summaries"][arm]["telemetry"]["forced_turn_order_controls"]
        for arm in evaluator.ARMS
    } == {arm: 500 for arm in evaluator.ARMS}
    assert receipt["comparisons"][
        f"recursive_rtp_minus_{evaluator.DIRECT_BRIDGE_ARM}"
    ]["opponent_seat_stratified"].keys() == {
        f"{opponent}|seat{seat}"
        for opponent in evaluator.R198_OFFICIAL_CONTROL_OPPONENTS
        for seat in (0, 1)
    }

    # The receipt is not merely adjacent to the snapshot-local registry: it
    # must bind both the physical JSON bytes and the V6 canonical slot digest.
    for field in (
        "matchup_adapter_registry_sha256",
        "matchup_adapter_slot_registry_digest",
    ):
        prepared_spec = json.loads(
            Path(str(fixture["materialized"]["prepared_evaluator_spec"]["path"])).read_text(
                encoding="utf-8"
            )
        )
        preflight_receipt = json.loads(
            Path(
                str(prepared_spec["shared_artifacts"]["planner_preflight_receipt"]["path"])
            ).read_text(encoding="utf-8")
        )
        preflight_receipt[field] = "sha256:" + "0" * 64
        bad_preflight = _write(
            tmp_path / f"bad-{field}.json", preflight_receipt
        )
        prepared_spec["shared_artifacts"]["planner_preflight_receipt"] = _identity(
            bad_preflight
        )
        with pytest.raises(
            RTPThreeArmEvaluationError,
            match=f"planner pass preflight mismatch at {field}",
        ):
            prepare_three_arm_manifest_from_spec(
                prepared_spec,
                output_path=tmp_path / f"bad-{field}-manifest.json",
            )

    bad_results = json.loads(results_path.read_text(encoding="utf-8"))
    bad_results["rows"][0]["rng_identity"] = {"requested_seed": 7}
    bad_path = _write(tmp_path / "bad-results.json", bad_results)
    with pytest.raises(RTPThreeArmEvaluationError, match="true RNG identity mismatch"):
        compile_three_arm_receipt(
            manifest_path=manifest_path,
            results=bad_path,
            output_path=tmp_path / "bad-receipt.json",
        )

    # The runner intentionally has no action-fence/action-execution authority
    # in arm A.  A synthetic fence must be rejected rather than normalized to
    # a sentinel digest, while the runner-shaped happy path above proves the
    # required A=None / B,C=SHA shape is accepted.
    fenced_results = json.loads(results_path.read_text(encoding="utf-8"))
    no_rtp_row = next(row for row in fenced_results["rows"] if row["arm"] == "no_rtp")
    source_execution = json.loads(
        Path(str(no_rtp_row["execution_receipt"]["path"])).read_text(encoding="utf-8")
    )
    source_execution["action_fence_sha256"] = _sha("forbidden-no-rtp-action-fence")
    source_execution["isolation"]["action_fence_sha256"] = source_execution[
        "action_fence_sha256"
    ]
    fenced_execution = _write(
        tmp_path / "bad-no-rtp-action-fence-execution.json", source_execution
    )
    no_rtp_row["execution_receipt"] = _identity(fenced_execution)
    fenced_path = _write(tmp_path / "bad-no-rtp-action-fence-results.json", fenced_results)
    with pytest.raises(RTPThreeArmEvaluationError, match="unexpectedly has an evaluator action fence"):
        compile_three_arm_receipt(
            manifest_path=manifest_path,
            results=fenced_path,
            output_path=tmp_path / "bad-no-rtp-action-fence-receipt.json",
        )

    prepared_spec = json.loads(
        Path(str(fixture["materialized"]["prepared_evaluator_spec"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    source_proof = json.loads(
        Path(str(prepared_spec["source_exclusion_proof"]["receipt"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    source_proof.pop("source_exclusion_computation")
    malformed_proof = _write(tmp_path / "malformed-source-proof.json", source_proof)
    prepared_spec["source_exclusion_proof"] = {"receipt": _identity(malformed_proof)}
    with pytest.raises(RTPThreeArmEvaluationError, match="source_exclusion_computation"):
        prepare_three_arm_manifest_from_spec(
            prepared_spec, output_path=tmp_path / "malformed-source-manifest.json"
        )

    closure_payload = json.loads(
        Path(str(prepared_spec["evaluation_cg_closure"]["receipt"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    parity = json.loads(Path(str(closure_payload["metadata_parity"]["path"])).read_text())
    parity.pop("distinct_dso_handles")
    malformed_parity = _write(tmp_path / "malformed-parity.json", parity)
    closure_payload["metadata_parity"] = _identity(malformed_parity)
    malformed_closure = _write(tmp_path / "malformed-closure.json", closure_payload)
    prepared_spec = json.loads(
        Path(str(fixture["materialized"]["prepared_evaluator_spec"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    prepared_spec["evaluation_cg_closure"] = {
        "receipt": _identity(malformed_closure),
        "runtime_library": prepared_spec["evaluation_cg_closure"]["runtime_library"],
    }
    with pytest.raises(RTPThreeArmEvaluationError, match="distinct_dso_handles"):
        prepare_three_arm_manifest_from_spec(
            prepared_spec, output_path=tmp_path / "malformed-closure-manifest.json"
        )


@pytest.mark.unit
def test_r198_fixed_contract_constants_are_not_test_fixture_defaults() -> None:
    assert evaluator.DIRECT_BRIDGE_ARM == "direct_bridge_recursive_disabled"
    assert evaluator.ARMS == (
        "no_rtp",
        "direct_bridge_recursive_disabled",
        "recursive_rtp",
    )
    assert evaluator.R198_MAX_NEURAL_PASSES == 256
    assert evaluator.R198_MAX_ACTION_COMBOS == 1024
    assert evaluator.OFFICIAL_CONTROL_PAIRED_CELLS == 1_000
    assert evaluator.MINIMUM_RECURSIVE_DECISIONS == 100
    assert evaluator.MINIMUM_RECURSIVE_INTENDED_COMPLEX_SHARE == 0.05
    assert evaluator.MAXIMUM_UNEXPECTED_RECURSIVE_FALLBACK_RATE == 0.01
