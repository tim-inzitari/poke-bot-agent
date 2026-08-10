"""Focused coverage for immutable r198 evaluation-input preparation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from poke_bot.engine_rebuild.rtp_pairing_snapshot import (
    PairingArtifactSet,
    emit_true_rng_pairing_capability,
    emit_true_rng_pairing_probe,
    frozen_file_identity,
    snapshot_abi_sha256,
)
from poke_bot.rtp_r198_evaluation_input_materializer import (
    CapturedSnapshot,
    R198EvaluationInputError,
    _r197_canonical_json_digest,
    materialize_r198_evaluation_inputs,
)
from poke_bot.rtp_r198_production_factory import r198_runtime_profile_payload


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: bytes | dict, *, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    path.chmod(mode)
    return path


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _capability(tmp_path: Path) -> Path:
    engine = _write(tmp_path / "engine.so", b"private engine", mode=0o555)
    source = _write(tmp_path / "source-manifest.json", {"schema": "source", "files": []})
    patch = _write(tmp_path / "patch.cpp", b"private patch")
    artifacts_before_receipt = PairingArtifactSet.from_paths(
        engine_path=engine,
        source_manifest_path=source,
        patch_path=patch,
        build_receipt_path=_write(tmp_path / "placeholder.json", {"placeholder": True}),
    )
    # Construct the build record after the three immutable input identities are
    # known; PairingArtifactSet is rebuilt with the final 0444 receipt below.
    build = _write(
        tmp_path / "build.json",
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
    del artifacts_before_receipt
    artifacts = PairingArtifactSet.from_paths(
        engine_path=engine,
        source_manifest_path=source,
        patch_path=patch,
        build_receipt_path=build,
    )
    probe = emit_true_rng_pairing_probe(
        output_path=tmp_path / "probe.json",
        artifacts=artifacts,
        deterministic_probe={
            "passed": True,
            "initial_snapshot_fingerprint_sha256": "sha256:" + "1" * 64,
            "initial_snapshot_fingerprint_bytes": 16,
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
    return emit_true_rng_pairing_capability(
        output_path=tmp_path / "capability.json",
        artifacts=artifacts,
        probe_path=probe,
    )


def _completion(tmp_path: Path) -> Path:
    train = ["r197-train-episode-a", "r197-train-episode-b"]
    heldout = ["r197-heldout-episode-a"]

    def side(name: str, rows: list[str]) -> dict:
        return {
            "candidate_selection": {
                "schema": "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1",
                "split": name,
            },
            "batch_cap_selection": {
                "retained_episode_count": len(rows),
                "retained_episode_ids_sha256": _r197_canonical_json_digest(rows),
                "row_level_sampling": False,
                "cross_window_dynamics_target": False,
            },
            "retained_episode_ids": rows,
        }

    selection = {
        "schema": "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1",
        "row_level_sampling": False,
        "cross_window_dynamics_target": False,
        "selection_plan_sha256": "sha256:" + "3" * 64,
        "train_selection_sha256": _r197_canonical_json_digest(train),
        "heldout_selection_sha256": _r197_canonical_json_digest(heldout),
        "train": side("train", train),
        "heldout": side("heldout", heldout),
    }
    return _write(
        tmp_path / "completion.json",
        {
            "schema": "poke_bot.alakazam_rtp_r197_shadow_candidate/v1",
            "status": "completed_shadow_only",
            "candidate_contract_sha256": "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e",
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
                    "manifest_sha256": "sha256:" + "4" * 64,
                    "receipt_sha256": "sha256:" + "5" * 64,
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


def _base_spec(tmp_path: Path) -> Path:
    candidate_deck = _write(tmp_path / "candidate.deck", ("\n".join(str(1000 + i) for i in range(60)) + "\n").encode())
    parent = _write(tmp_path / "parent.pt", b"parent")
    matchup = _write(tmp_path / "matchup.json", {"tree": "frozen"})
    sidecar = _write(tmp_path / "sidecar.pt", b"sidecar")
    matchup_adapter_registry = _write(
        tmp_path / "state" / "matchup_adapter_roster.json",
        (ROOT / "state" / "matchup_adapter_roster.json").read_bytes(),
    )
    arms: dict[str, dict] = {}
    for arm in ("no_rtp", "direct_bridge_recursive_disabled", "recursive_rtp"):
        runtime = _write(tmp_path / f"{arm}.runtime", arm.encode())
        arms[arm] = {
            "runtime_artifact": _identity(runtime),
            "runtime_profile_payload": r198_runtime_profile_payload(arm),
        }
        if arm != "no_rtp":
            arms[arm]["rtp_sidecar"] = _identity(sidecar)
    panel = (
        ("iono", "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2"),
        ("dragapult-ex", "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0"),
        ("mega-abomasnow-ex", "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb"),
        ("mega-lucario-ex", "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a"),
    )
    opponents = []
    for index, (identifier, content_digest) in enumerate(panel):
        deck = _write(tmp_path / f"{identifier}.deck", ("\n".join(str(2000 + index * 100 + i) for i in range(60)) + "\n").encode())
        opponents.append(
            {
                "id": identifier,
                "content_digest": content_digest,
                "deck": _identity(deck),
                "artifact": _identity(_write(tmp_path / f"{identifier}.package.json", {"sealed": True})),
                "package_root": str(tmp_path),
            }
        )
    cg_closure = _write(
        tmp_path / "evaluation-cg-closure.json",
        {"schema": "test-evaluation-cg-closure", "status": "sealed"},
    )
    runtime_library = _write(
        tmp_path / "snapshot-eval-cg" / "cg" / "libcg.so",
        b"snapshot-local test cg engine",
        mode=0o444,
    )
    return _write(
        tmp_path / "base-spec.json",
        {
            "shared_artifacts": {
                "parent_checkpoint": _identity(parent),
                "deck": _identity(candidate_deck),
                "matchup_tree": _identity(matchup),
            },
            "arms": arms,
            "opponents": opponents,
            "production_factory": {
                "source_snapshot_root": str(tmp_path),
                "artifacts": {"deck": _identity(candidate_deck)},
                "evaluation_cg": {"library": _identity(runtime_library)},
                "matchup_adapter_registry": {
                    **_identity(matchup_adapter_registry),
                    "mode": 0o444,
                },
            },
            "candidate_evaluation_binding": {
                "schema": "poke_bot.recursive_turn_planner.r198_candidate_evaluation_binding/v1",
                "status": "bound",
                "candidate_contract_sha256": "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e",
                "parent_checkpoint_sha256": "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a",
                "sidecar_sha256": "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31",
                "sidecar_config_sha256": "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676",
                "deck_file_sha256": "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65",
                "deck_cards_sha256": "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247",
                "matchup_tree_sha256": "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049",
                "sizing_profile": "pure_rl_r197",
                "max_neural_passes": 256,
                "max_action_combos": 1024,
                "required_neural_passes": {"normal": 6, "forced_replan": 5},
            },
            "evaluation_cg_closure": {
                "receipt": _identity(cg_closure),
                "runtime_library": _identity(runtime_library),
            },
        },
    )


@pytest.mark.unit
def test_materializer_seals_exact_official_panel_and_computes_source_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `/var` is a symlink to `/private/var` on macOS.  The materializer
    # intentionally rejects symlinked evidence paths, so use the physical
    # pytest directory in this focused test.
    tmp_path = Path(os.path.realpath(tmp_path))
    capability = _capability(tmp_path / "pairing")
    completion = _completion(tmp_path / "candidate")
    registry = _write(
        tmp_path / "registry.json",
        (ROOT / "ops" / "research_control_registry_v1.json").read_bytes(),
    )
    base_spec = _base_spec(tmp_path / "base")
    capture_count = 0

    def capture(_deck0: object, _deck1: object, _seed: int) -> CapturedSnapshot:
        nonlocal capture_count
        payload = f"opaque-native-snapshot-{capture_count}".encode()
        capture_count += 1
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        return CapturedSnapshot(payload, f"snapshot-{digest[7:31]}", digest, len(payload))

    def preflight(preflight_input: dict, output_path: Path) -> Path:
        direct = preflight_input["arms"]["direct_bridge_recursive_disabled"]
        recursive = preflight_input["arms"]["recursive_rtp"]
        sidecar = direct["rtp_sidecar"]
        assert preflight_input["fixtures"]["normal"]["snapshot_seal"]
        assert preflight_input["fixtures"]["forced_replan"]["snapshot_seal"]
        assert preflight_input["fixtures"]["normal"]["observation"]
        assert preflight_input["fixtures"]["forced_replan"]["observation"]
        assert preflight_input["fixtures"]["normal"]["candidate_deck"]
        assert preflight_input["fixtures"]["normal"]["opponent_deck"]
        assert preflight_input["fixtures"]["normal"]["candidate_seat"] == 0
        assert stat.S_IMODE(
            Path(preflight_input["production_factory"]["evaluation_inputs_root"]).stat().st_mode
        ) == 0o555
        assert preflight_input["production_factory"]["matchup_adapter_registry"] == {
            "path": str(base_spec.parent / "state" / "matchup_adapter_roster.json"),
            "sha256": "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc",
            "bytes": 11_899,
            "mode": 0o444,
        }
        _write(
            output_path,
            {
                "schema": "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1",
                "status": "passed",
                "sidecar_sha256": sidecar["sha256"],
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
        return output_path

    def fake_prepare(spec: dict, *, output_path: Path) -> Path:
        assert len(spec["rng_materials"]) == 1000
        assert "artifact" not in spec["rng_materials"][0]
        assert spec["candidate_evaluation_binding"]["max_neural_passes"] == 256
        assert set(spec["evaluation_cg_closure"]) == {"receipt", "runtime_library"}
        assert spec["production_factory"]["matchup_adapter_registry"] == {
            "path": str(base_spec.parent / "state" / "matchup_adapter_roster.json"),
            "sha256": "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc",
            "bytes": 11_899,
            "mode": 0o444,
        }
        payload = {
            "schema": "poke_bot.recursive_turn_planner.three_arm_evaluation_manifest/v2",
            "status": "prepared_true_rng_pairing_required",
            "manifest_input_sha256": "sha256:" + "a" * 64,
        }
        _write(output_path, payload)
        return output_path

    import poke_bot.rtp_three_arm_evaluation as evaluator

    monkeypatch.setattr(evaluator, "prepare_three_arm_manifest_from_spec", fake_prepare)
    seeds = iter(range(50_000, 52_000))
    result = materialize_r198_evaluation_inputs(
        completion_receipt=completion,
        research_control_registry=registry,
        pairing_capability=capability,
        evaluator_base_spec=base_spec,
        output_root=tmp_path / "out",
        run_nonce="unit-materializer",
        snapshot_capturer=capture,
        preflight_runner=preflight,
        seed_provider=lambda: next(seeds),
        fixture_observation_extractor=lambda _seal: {
            "current": {"result": -1},
            "select": {"minCount": 1, "maxCount": 1},
        },
    )

    assert result["paired_cell_count"] == 1000
    assert capture_count == 1002  # 1,000 scored cells + 2 unscored preflight fixtures
    materials = json.loads(Path(result["rng_materials_manifest"]["path"]).read_text())
    assert len(materials["rng_materials"]) == 1000
    assert materials["paired_cell_count"] == 1000
    first = materials["rng_materials"][0]
    assert set(first) == {
        "id",
        "kind",
        "snapshot_artifact",
        "seal",
        "opponent_id",
        "candidate_seat",
        "replicate",
        "evaluation_case_id",
        "requested_seed_audit_only",
    }
    assert stat.S_IMODE(Path(first["snapshot_artifact"]["path"]).stat().st_mode) == 0o444
    assert stat.S_IMODE(Path(first["seal"]["path"]).stat().st_mode) == 0o444
    seal = json.loads(Path(first["seal"]["path"]).read_text())
    binding_path = Path(seal["case_binding_artifact"]["path"])
    binding = json.loads(binding_path.read_text())
    assert stat.S_IMODE(binding_path.stat().st_mode) == 0o444
    assert seal["snapshot_id"] == first["id"]
    assert seal["rng_kind"] == "snapshot"
    assert seal["requested_seed_is_pairing_proof"] is False
    assert seal["candidate_seat"] == first["candidate_seat"]
    assert seal["evaluation_only"] is True
    assert seal["training_eligible"] is False
    assert binding["debug_seed"] == first["requested_seed_audit_only"]
    assert binding["cell_id"] == "cell-000000"
    assert binding["evaluation_case_bindings_sha256"]
    for fixture in materials["preflight_fixtures"].values():
        observation_path = Path(fixture["observation"]["path"])
        assert stat.S_IMODE(observation_path.stat().st_mode) == 0o444
        assert fixture["expected_mode"] in {"recursive_plan", "forced_replan"}
    proof = json.loads(Path(result["source_exclusion_proof"]["path"]).read_text())
    computation = proof["source_exclusion_computation"]
    assert computation["method"] == "exact_source_id_set_intersection"
    assert computation["r197_train_episode_count"] == 2
    assert computation["r197_heldout_episode_count"] == 1
    assert computation["evaluation_case_source_count"] == 1000
    assert computation["intersection_episode_count"] == 0
    prepared_spec = json.loads(
        Path(result["prepared_evaluator_spec"]["path"]).read_text(encoding="utf-8")
    )
    assert prepared_spec["production_factory"]["matchup_adapter_registry"] == {
        "path": str(base_spec.parent / "state" / "matchup_adapter_roster.json"),
        "sha256": "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc",
        "bytes": 11_899,
        "mode": 0o444,
    }
    authority = json.loads(Path(result["evaluation_only_authority"]["path"]).read_text())
    assert authority == {
        "schema": "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1",
        "status": "authorized_evaluation_only",
        "manifest_sha256": result["prepared_evaluator_manifest"]["sha256"],
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
        "record_sha256": authority["record_sha256"],
    }
    with pytest.raises(R198EvaluationInputError, match="refusing to reuse"):
        materialize_r198_evaluation_inputs(
            completion_receipt=completion,
            research_control_registry=registry,
            pairing_capability=capability,
            evaluator_base_spec=base_spec,
            output_root=tmp_path / "out",
            run_nonce="unit-materializer",
            snapshot_capturer=capture,
            preflight_runner=preflight,
            seed_provider=lambda: 1,
        )


@pytest.mark.unit
def test_materializer_rejects_duplicate_debug_seeds_before_second_capture(tmp_path: Path) -> None:
    tmp_path = Path(os.path.realpath(tmp_path))
    capability = _capability(tmp_path / "pairing")
    completion = _completion(tmp_path / "candidate")
    registry = _write(
        tmp_path / "registry.json",
        (ROOT / "ops" / "research_control_registry_v1.json").read_bytes(),
    )
    base_spec = _base_spec(tmp_path / "base")
    captures = 0

    def capture(_deck0: object, _deck1: object, _seed: int) -> CapturedSnapshot:
        nonlocal captures
        payload = f"duplicate-seed-check-{captures}".encode()
        captures += 1
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        return CapturedSnapshot(payload, f"snapshot-{digest[7:31]}", digest, len(payload))

    with pytest.raises(R198EvaluationInputError, match="globally unique"):
        materialize_r198_evaluation_inputs(
            completion_receipt=completion,
            research_control_registry=registry,
            pairing_capability=capability,
            evaluator_base_spec=base_spec,
            output_root=tmp_path / "out",
            run_nonce="duplicate-seed",
            snapshot_capturer=capture,
            fixture_observation_extractor=lambda _seal: {"current": {"result": -1}},
            seed_provider=lambda: 7,
        )
    assert captures == 1
    frozen = list((tmp_path / "out").glob("r198-evaluation-inputs-*"))
    assert len(frozen) == 1
    assert stat.S_IMODE(frozen[0].stat().st_mode) == 0o555


@pytest.mark.unit
def test_materializer_rejects_writable_snapshot_local_roster_before_capture(
    tmp_path: Path,
) -> None:
    tmp_path = Path(os.path.realpath(tmp_path))
    capability = _capability(tmp_path / "pairing")
    completion = _completion(tmp_path / "candidate")
    registry = _write(
        tmp_path / "registry.json",
        (ROOT / "ops" / "research_control_registry_v1.json").read_bytes(),
    )
    base_spec = _base_spec(tmp_path / "base")
    roster = base_spec.parent / "state" / "matchup_adapter_roster.json"
    roster.chmod(0o644)
    captures = 0

    def capture(_deck0: object, _deck1: object, _seed: int) -> CapturedSnapshot:
        nonlocal captures
        captures += 1
        raise AssertionError("the roster guard must fail before snapshot capture")

    with pytest.raises(R198EvaluationInputError, match="must be immutable"):
        materialize_r198_evaluation_inputs(
            completion_receipt=completion,
            research_control_registry=registry,
            pairing_capability=capability,
            evaluator_base_spec=base_spec,
            output_root=tmp_path / "out",
            run_nonce="writable-roster",
            snapshot_capturer=capture,
        )
    assert captures == 0
    assert not (tmp_path / "out").exists()
