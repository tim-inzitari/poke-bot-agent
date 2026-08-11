"""CPU-only contract coverage for the sealed r244 smoke/replay harnesses."""

from __future__ import annotations

import importlib
import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r228_kaggle_r244_harness_common as common


def _load_script(name: str):
    path = SCRIPTS / name
    # Avoid colliding with the real test module for the immutable binding
    # builder when this harness suite is collected together with it.
    spec = importlib.util.spec_from_file_location(
        f"r244_smoke_harness_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SMOKE = _load_script("run_r228_async_eight_worker_packaged_smoke.py")
REPLAY = _load_script("replay_r228_kaggle_failure_91766923.py")
BUILDER = _load_script("build_r235_r236_immutable_replacement_binding.py")


def _exact_stage_owned_modules() -> dict[str, object]:
    """Snapshot the cache entries an exact-stage loader is allowed to evict."""

    return {
        name: module
        for name, module in sys.modules.items()
        if name in {"main", "r195_direct_main", "r228_r195_direct"}
        or name == "poke_bot"
        or name.startswith("poke_bot.")
    }


def _clear_exact_stage_owned_modules() -> None:
    for name in tuple(sys.modules):
        if (
            name in {"main", "r195_direct_main", "r228_r195_direct"}
            or name == "poke_bot"
            or name.startswith("poke_bot.")
        ):
            sys.modules.pop(name, None)


def _write_import_fixture(
    root: Path, *, origin: str, stage_main: bool, include_broker: bool
) -> None:
    """Build a tiny package whose origin is obvious in a cache-leak test."""

    package = root / "poke_bot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"ORIGIN = {origin!r}\n", encoding="utf-8")
    (package / "features.py").write_text(
        f"ORIGIN = {origin!r}\n", encoding="utf-8"
    )
    (package / "cg_env.py").write_text(
        f"ORIGIN = {origin!r}\n", encoding="utf-8"
    )
    if include_broker:
        (package / "r228_kaggle_broker.py").write_text(
            f"ORIGIN = {origin!r}\n", encoding="utf-8"
        )
    if stage_main:
        main_source = (
            "from poke_bot import features\n"
            "from poke_bot.r228_kaggle_broker import ORIGIN as BROKER_ORIGIN\n"
            "ORIGIN = features.ORIGIN\n"
            "def _direct():\n"
            "    return {'origin': ORIGIN, 'broker_origin': BROKER_ORIGIN}\n"
        )
    else:
        # The deliberately incomplete preloaded package reproduces the Train
        # failure: a cached host ``poke_bot`` has no staged broker submodule.
        main_source = (
            f"ORIGIN = {origin!r}\n"
            "def _direct():\n"
            "    return {'origin': ORIGIN}\n"
        )
    (root / "main.py").write_text(main_source, encoding="utf-8")


@pytest.mark.parametrize("harness_name", ("replay", "smoke"))
def test_exact_stage_load_evicts_preloaded_wrong_poke_bot(
    tmp_path: Path, harness_name: str
) -> None:
    """A host-imported package cannot shadow the immutable extracted stage."""

    wrong_root = tmp_path / "host-repository"
    stage = tmp_path / "sealed-stage"
    _write_import_fixture(
        wrong_root, origin="host-repository", stage_main=False, include_broker=False
    )
    _write_import_fixture(stage, origin="sealed-stage", stage_main=True, include_broker=True)

    original_modules = _exact_stage_owned_modules()
    original_path = list(sys.path)
    original_cwd = Path.cwd()
    original_cg_lib_path = os.environ.get("CG_LIB_PATH")
    original_dont_write_bytecode = sys.dont_write_bytecode
    try:
        _clear_exact_stage_owned_modules()
        sys.path.insert(0, str(wrong_root))
        wrong_package = importlib.import_module("poke_bot")
        wrong_features = importlib.import_module("poke_bot.features")
        wrong_main = importlib.import_module("main")
        assert wrong_package.ORIGIN == "host-repository"
        assert wrong_features.ORIGIN == "host-repository"
        assert wrong_main.ORIGIN == "host-repository"
        assert "poke_bot.r228_kaggle_broker" not in sys.modules

        loader = REPLAY._load_stage if harness_name == "replay" else SMOKE._load_stage
        main, second, features = loader(stage)

        assert main.ORIGIN == "sealed-stage"
        assert main.BROKER_ORIGIN == "sealed-stage"
        assert features.ORIGIN == "sealed-stage"
        assert sys.modules["poke_bot"] is not wrong_package
        assert sys.modules["poke_bot.features"] is not wrong_features
        assert Path(sys.modules["poke_bot"].__file__).resolve().is_relative_to(
            stage.resolve()
        )
        assert Path(sys.modules["poke_bot.r228_kaggle_broker"].__file__).resolve().is_relative_to(
            stage.resolve()
        )
        if harness_name == "replay":
            assert second == {"origin": "sealed-stage", "broker_origin": "sealed-stage"}
        else:
            assert second.ORIGIN == "sealed-stage"
    finally:
        _clear_exact_stage_owned_modules()
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        os.chdir(original_cwd)
        if original_cg_lib_path is None:
            os.environ.pop("CG_LIB_PATH", None)
        else:
            os.environ["CG_LIB_PATH"] = original_cg_lib_path
        sys.dont_write_bytecode = original_dont_write_bytecode


def _base_lane_receipt() -> dict[str, object]:
    return {
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_end_calls": 2,
        "search_release_calls": 2,
        "distinct_search_begin_composite_count": 2,
        "per_lane_handle_identities": ["handle-a", "handle-b"],
        # Official libcg's first native SearchId is handle-local, so these are
        # intentionally equal raw integers.
        "per_lane_first_search_ids": [0, 0],
        "per_lane_search_id_chains": [[0], [0]],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": "handle-a", "first_search_id": 0},
            {"lane_id": 1, "handle_identity": "handle-b", "first_search_id": 0},
        ],
        "per_lane_depth": [4, 4],
        "search_step_calls": 2,
        "outstanding_virtual_loss": 0,
    }


def _cpu_cuda_observation() -> dict[str, object]:
    """A valid observation, not a resource-envelope claim about CUDA."""

    return {
        "schema": common.CUDA_RUNTIME_OBSERVATION_SCHEMA,
        "phase": common.CUDA_RUNTIME_OBSERVATION_PHASE,
        "torch_imported": True,
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "devices": [],
        "model_device": "cpu",
        "telemetry_complete": True,
        "error_types": [],
    }


def _base_decision(mode: str = "shared_tree_mcts") -> dict[str, object]:
    marker: dict[str, object] = {
        "mode": mode,
        "selected_action": [0],
        "direct_action": [1],
        "complete_action_cap": 65_536,
        "configured_simulator_lane_count": 2,
        "legal_action_count": 2,
        "root_visits": 8,
        "parent_cuda_runtime_before_search": _cpu_cuda_observation(),
    }
    marker.update(_base_lane_receipt())
    return marker


def test_r244_accepts_handle_local_duplicate_raw_search_ids() -> None:
    marker = _base_decision()
    marker.update(
        {
            "mcts_action_authority": True,
            "confidence_classification": "ambiguous_mcts",
            "mcts_child_started": True,
            "mcts_child_call_count": 1,
            "child_search_hard_seconds": 2.0,
            "parent_action_hard_seconds": 4.0,
            "all_selected_factorized_stages_meet_threshold": False,
            "completed_backups": 8,
            "selected_action_visits": 8,
            "both_lanes_progressed": True,
            "max_simulator_calls_in_flight": 2,
            "microbatch_sizes": [2, 2, 2, 2],
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations_required": 3,
            "maximum_backups_per_decision": 32,
            "stop_reason": "stable_root_leader",
            "deterministic_root_leader_observations": 3,
            "actor_change_boundary_leaf_count": 1,
            "chance_boundary_leaf_count": 1,
            "boundary_leaf_count": 1,
            "broker": {
                "child_identity": {
                    "cuda_runtime_before_search": _cpu_cuda_observation(),
                }
            },
        }
    )
    result = common.validate_decision_marker(marker, legal_actions=[[0], [1]])
    assert result["mode"] == "shared_tree_mcts"
    assert result["lane_receipt"]["first_search_ids"] == [0, 0]


def test_r244_rejects_duplicate_handle_search_composite() -> None:
    marker = _base_decision()
    marker.update(
        {
            "mcts_action_authority": True,
            "confidence_classification": "ambiguous_mcts",
            "mcts_child_started": True,
            "mcts_child_call_count": 1,
            "child_search_hard_seconds": 2.0,
            "parent_action_hard_seconds": 4.0,
            "all_selected_factorized_stages_meet_threshold": False,
            "completed_backups": 8,
            "selected_action_visits": 8,
            "both_lanes_progressed": True,
            "max_simulator_calls_in_flight": 2,
            "microbatch_sizes": [2, 2, 2, 2],
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations_required": 3,
            "maximum_backups_per_decision": 32,
            "stop_reason": "stable_root_leader",
            "deterministic_root_leader_observations": 3,
            "actor_change_boundary_leaf_count": 0,
            "chance_boundary_leaf_count": 0,
            "boundary_leaf_count": 0,
            "broker": {
                "child_identity": {
                    "cuda_runtime_before_search": _cpu_cuda_observation(),
                }
            },
            "per_lane_handle_identities": ["same", "same"],
            "handle_scoped_first_search_id_composite_states": [
                {"lane_id": 0, "handle_identity": "same", "first_search_id": 0},
                {"lane_id": 1, "handle_identity": "same", "first_search_id": 0},
            ],
        }
    )
    with pytest.raises(common.HarnessContractError, match="distinct handles"):
        common.validate_decision_marker(marker, legal_actions=[[0], [1]])


def test_high_confidence_accepts_inclusive_point_eight_without_child() -> None:
    marker = _base_decision("high_confidence_frozen_direct")
    marker.update(
        {
            "selected_action": [1],
            "direct_action": [1],
            "mcts_action_authority": False,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count": 1,
            "degraded": False,
            "selected_factorized_stage_probability_threshold": 0.80,
            "selected_factorized_stage_probabilities": [0.8, 0.99],
            "all_selected_factorized_stages_meet_threshold": True,
        }
    )
    result = common.validate_decision_marker(marker, legal_actions=[[0], [1]])
    assert result["mode"] == "high_confidence_frozen_direct"


def test_cuda_observation_accepts_a_real_gpu_without_resource_inference() -> None:
    observation = _cpu_cuda_observation()
    observation.update(
        {
            "cuda_available": True,
            "cuda_initialized": True,
            "device_count": 1,
            "devices": [
                {
                    "device_index": 0,
                    "device_name": "NVIDIA Test GPU",
                    "total_memory_bytes": 16 * 1024**3,
                    "free_memory_bytes": 12 * 1024**3,
                }
            ],
            "model_device": "cuda:0",
        }
    )
    validated = common.validate_cuda_runtime_observation(
        observation, field="parent_cuda_runtime_before_search"
    )
    assert validated["cuda_available"] is True
    assert validated["device_count"] == 1


def test_replay_harness_does_not_override_cuda_visibility() -> None:
    source = (SCRIPTS / "replay_r228_kaggle_failure_91766923.py").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "cuda_visible_devices" not in source


def test_r246_harness_pin_matches_the_canonical_r225_source_and_binder() -> None:
    contract = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    assert common.sha256_file(contract) == common.R246_R225_TYPED_CONTRACT_SHA256
    assert (
        BUILDER.CANONICAL_R225_R240_TYPED_CONTRACT_SHA256
        == common.R246_R225_TYPED_CONTRACT_SHA256
    )


def test_r246_accepts_only_current_deterministic_terminal_win_proof() -> None:
    observation = {"step": 58, "current": {"yourIndex": 0}, "select": {"option": []}}
    legal = [[0], [1]]
    marker = _base_decision()
    root_observation = common.canonical_observation_fingerprint(observation)
    root_legal = common.legal_order_fingerprint(legal)
    marker.update(
        {
            "mcts_action_authority": True,
            "confidence_classification": "ambiguous_mcts",
            "mcts_child_started": True,
            "mcts_child_call_count": 1,
            "child_search_hard_seconds": 2.0,
            "parent_action_hard_seconds": 4.0,
            "all_selected_factorized_stages_meet_threshold": False,
            "completed_backups": 1,
            "selected_action_visits": 1,
            "max_simulator_calls_in_flight": 2,
            "microbatch_sizes": [1],
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations_required": 3,
            "maximum_backups_per_decision": 32,
            "stop_reason": common.PROVEN_TERMINAL_WIN_STOP_REASON,
            "deterministic_root_leader_observations": 0,
            "actor_change_boundary_leaf_count": 0,
            "chance_boundary_leaf_count": 0,
            "boundary_leaf_count": 0,
            "search_step_calls": 2,
            "per_lane_depth": [1, 0],
            "root_visits": 1,
            "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
            "proven_deterministic_terminal_win_this_turn": True,
            "root_observation_fingerprint": root_observation,
            "root_legal_order_fingerprint": root_legal,
            "legal_order_fingerprint": root_legal,
            "root_actor_seat": 0,
            "principal_variation": [],
            "terminal_win_proof": {
                "proof_kind": common.PROVEN_TERMINAL_WIN_PROOF_KIND,
                "root_observation_fingerprint": root_observation,
                "root_legal_order_fingerprint": root_legal,
                "root_actor_seat": 0,
                "root_action": [0],
                "selected_action": [0],
                "terminal_result": "win",
                "terminal_winner_seat": 0,
                "terminal_leaf_reached": True,
                "proof_path_action_count": 1,
                "discovering_lane_id": 0,
                "path_actor_seats": [0],
                "path_no_chance_boundary": True,
                "path_no_actor_change_boundary": True,
                "path_no_opponent_boundary_crossing": True,
                "path_no_unresolved_randomness": True,
                "proof_is_deterministic": True,
            },
            "broker": {
                "child_identity": {
                    "cuda_runtime_before_search": _cpu_cuda_observation(),
                }
            },
        }
    )
    result = common.validate_decision_marker(
        marker, legal_actions=legal, observation=observation
    )
    assert result["terminal_win_proof"]["terminal_result"] == "win"
    marker["terminal_win_proof"]["path_no_chance_boundary"] = False
    with pytest.raises(common.HarnessContractError, match="path_no_chance_boundary"):
        common.validate_decision_marker(marker, legal_actions=legal, observation=observation)


def test_normal_mcts_requires_two_backups_and_exact_depth_accounting() -> None:
    marker = _base_decision()
    marker.update(
        {
            "mcts_action_authority": True,
            "confidence_classification": "ambiguous_mcts",
            "mcts_child_started": True,
            "mcts_child_call_count": 1,
            "child_search_hard_seconds": 2.0,
            "parent_action_hard_seconds": 4.0,
            "all_selected_factorized_stages_meet_threshold": False,
            "completed_backups": 1,
            "selected_action_visits": 1,
            "both_lanes_progressed": True,
            "max_simulator_calls_in_flight": 2,
            "microbatch_sizes": [1],
            "minimum_backups_before_stability": 8,
            "stable_root_leader_observations_required": 3,
            "maximum_backups_per_decision": 32,
            "stop_reason": "decision_deadline",
            "deterministic_root_leader_observations": 0,
            "actor_change_boundary_leaf_count": 0,
            "chance_boundary_leaf_count": 0,
            "boundary_leaf_count": 0,
            "broker": {
                "child_identity": {
                    "cuda_runtime_before_search": _cpu_cuda_observation(),
                }
            },
        }
    )
    with pytest.raises(common.HarnessContractError, match="completed backups"):
        common.validate_decision_marker(marker, legal_actions=[[0], [1]])


def test_clean_zero_requires_exact_reap_and_two_lane_receipt() -> None:
    marker = _base_decision("zero_backup_precomputed_direct_fallback")
    marker.update(
        {
            "selected_action": [1],
            "direct_action": [1],
            "mcts_action_authority": False,
            "zero_backup_precomputed_direct_fallback": True,
            "clean_deadline_cleanup_complete": True,
            "completed_backups": 0,
            "stop_reason": "decision_deadline",
            "search_step_calls": 2,
            "max_simulator_calls_in_flight": 2,
            "microbatch_sizes": [],
            "per_lane_depth": [0, 0],
            "exact_child_cleanup_and_reap": {"reap": {"reaped": True}},
            "broker": {
                "child_identity": {
                    "cuda_runtime_before_search": _cpu_cuda_observation(),
                }
            },
        }
    )
    result = common.validate_decision_marker(marker, legal_actions=[[0], [1]])
    assert result["clean_zero_reap"]["reaped"] is True


def test_degraded_marker_requires_package_child_reap() -> None:
    marker = {
        "selected_action": [0],
        "direct_action": [0],
        "mcts_action_authority": False,
        "action_authority": "precomputed_frozen_r195_direct_action",
        "complete_action_cap": 65_536,
        "configured_simulator_lane_count": 2,
        "parent_cuda_runtime_before_search": _cpu_cuda_observation(),
        "per_lane_progress": {"0": {"phase": "native_step"}},
        "child_fault": {
            "configured_simulator_lane_count": 2,
            "code": "response_timeout",
            "child_reap": {"child_present": True, "reaped": True},
        },
    }
    result = common.validate_degraded_marker(marker, legal_actions=[[0], [1]])
    assert result["fault_code"] == "response_timeout"


def test_smoke_disallows_terminal_success_before_true_terminal() -> None:
    marker = _base_decision("high_confidence_frozen_direct")
    marker.update(
        {
            "selected_action": [1],
            "direct_action": [1],
            "mcts_action_authority": False,
            "mcts_child_started_for_this_decision": False,
            "mcts_select_call_count": 0,
            "history_only_existing_child_journal_count": 0,
            "degraded": False,
            "selected_factorized_stage_probability_threshold": 0.8,
            "selected_factorized_stage_probabilities": [0.8],
            "all_selected_factorized_stages_meet_threshold": True,
        }
    )
    with pytest.raises(common.HarnessContractError, match="before terminal"):
        SMOKE._validate_callback_markers(
            legal=[[0], [1]],
            observation={"step": 1},
            callback_markers={
                "decisions": [marker],
                "degraded_fallbacks": [],
                "hard_failures": [],
                "full_gameplay_successes": [{"unexpected": True}],
            },
            previous_degraded_count=0,
        )


def test_raw_probe_callback_keeps_the_actual_authority_marker_or_null() -> None:
    marker = _base_decision("high_confidence_frozen_direct")
    callback = {
        "call_index": 0,
        "decision_marker_or_containment": marker,
        "raw_decision_markers": [marker],
        "raw_containment_markers": [],
    }
    payload = {
        "status": "passed",
        "stock_game": {"actions": [callback]},
        "markers": {
            "decisions": [marker],
            "degraded_fallbacks": [],
            "hard_failures": [],
            "full_gameplay_successes": [],
        },
    }
    journal = SMOKE._Journal()
    envelope = SMOKE._raw_r240_probe_envelope(
        payload=payload, journal=journal, receipt=Path("/tmp/raw-r240-receipt.json")
    )
    assert envelope["callbacks"][0]["decision_marker_or_containment"] == marker


def test_saved_replay_is_exact_seat0_step58_two_choice_history() -> None:
    target, steps = REPLAY._load_replay_target(REPLAY.DEFAULT_REPLAY, 58)
    events = REPLAY._prior_active_events(steps, 58)
    assert target["step"] == 58
    assert target["current"]["yourIndex"] == 0
    assert len(target["select"]["option"]) == 2
    assert events
    assert all(event["replay_step_index"] < 58 for event in events)


def test_full_game_success_requires_non_degraded_two_lane_terminal_receipt() -> None:
    marker = {
        "configured_simulator_lane_count": 2,
        "complete_action_cap": 65_536,
        "degraded_fault_count": 0,
        "mcts_branching_decisions": 3,
        "parent_cuda_runtime_before_search": _cpu_cuda_observation(),
    }
    common.validate_full_game_success(marker, mcts_decision_count=3)
    marker["degraded_fault_count"] = 1
    with pytest.raises(common.HarnessContractError, match="degraded"):
        common.validate_full_game_success(marker, mcts_decision_count=3)


def test_saved_and_full_game_receipt_shapes_are_accepted_by_r235_binder() -> None:
    """The two executable harnesses emit the binder's common gate shape."""

    identity = {
        "candidate_archive_sha256": "sha256:" + "a" * 64,
        "candidate_archive_size_bytes": 123,
        "member_manifest_sha256": "sha256:" + "b" * 64,
        "entrypoint_sha256": "sha256:" + "c" * 64,
        "r225_contract_sha256": "sha256:" + "d" * 64,
        "canonical_libcg_contract_sha256": "sha256:" + "e" * 64,
        "linux_x86_64_libcg_sha256": "sha256:" + "f" * 64,
        "linux_x86_64_libcg_size_bytes": 1_342_400,
        "complete_ordered_action_cap": 65_536,
        "simulator_search_lane_count": 2,
        "phase1_submission_environment": dict(common.PHASE1_SUBMISSION_ENVIRONMENT),
        "r240_hybrid_scheduler": dict(common.R242_BINDING_SCHEDULER),
        "deterministic_continuation": dict(common.BINDING_DETERMINISTIC_CONTINUATION),
    }
    assert identity["r240_hybrid_scheduler"] == BUILDER.R242_HYBRID_SCHEDULER
    assert identity["deterministic_continuation"] == BUILDER.DETERMINISTIC_CONTINUATION

    saved = common.passed_preflight_receipt(
        receipt_name=common.SAVED_EPISODE_RECEIPT_NAME,
        common_identity=identity,
        harness_schema=REPLAY.SCHEMA,
    )
    saved.update(
        {
            "source_submission_id": 55_416_396,
            "source_episode_id": 91_766_923,
            "seat": 0,
            "final_callback_step": 58,
            "final_callback_ordered_legal_action_count": 2,
            "legal_action_before_hard_deadline": True,
            "fault_injected_broker_child_reap_proved": True,
            "result_path": "contained_precomputed_parent_direct_fallback_after_exact_child_reap",
        }
    )
    BUILDER._receipt_common(
        saved, receipt_name=common.SAVED_EPISODE_RECEIPT_NAME, identity=identity
    )
    BUILDER._validate_saved_episode_receipt(saved, identity)

    full = common.passed_preflight_receipt(
        receipt_name=common.FULL_GAME_RECEIPT_NAME,
        common_identity=identity,
        harness_schema=SMOKE.SCHEMA,
    )
    full.update(
        {
            "exact_package_full_local_game_passed": True,
            "full_gameplay_loop_completed": True,
            "branching_gameplay_decision_count": 1,
            "explicit_success_marker_count": 1,
            "degraded_game_count": 0,
            "active_simulator_search_lane_count": 2,
        }
    )
    BUILDER._receipt_common(
        full, receipt_name=common.FULL_GAME_RECEIPT_NAME, identity=identity
    )
    BUILDER._validate_full_game_receipt(full, identity)


@pytest.mark.parametrize(
    ("writer", "payload"),
    [
        (REPLAY._write_receipt_once, {"saved_replay": True}),
        (SMOKE._write_receipt_once, {"full_game": True}),
    ],
    ids=("saved-replay", "full-game"),
)
def test_saved_and_full_game_receipt_writers_publish_mode_0444(
    tmp_path: Path, writer: object, payload: dict[str, object]
) -> None:
    """Final local-gate paths are sealed before their hard-link publication."""

    assert callable(writer)
    receipt = tmp_path / "immutable-local-gate.json"
    writer(receipt, payload)

    assert receipt.is_file() and not receipt.is_symlink()
    assert stat.S_IMODE(os.lstat(receipt).st_mode) == 0o444


def test_r246_terminal_win_receipt_shape_is_accepted_by_r235_binder() -> None:
    identity = {
        "candidate_archive_sha256": "sha256:" + "a" * 64,
        "candidate_archive_size_bytes": 123,
        "member_manifest_sha256": "sha256:" + "b" * 64,
        "entrypoint_sha256": "sha256:" + "c" * 64,
        "r225_contract_sha256": common.R246_R225_TYPED_CONTRACT_SHA256,
        "canonical_libcg_contract_sha256": "sha256:" + "e" * 64,
        "linux_x86_64_libcg_sha256": "sha256:" + "f" * 64,
        "linux_x86_64_libcg_size_bytes": 1_342_400,
        "complete_ordered_action_cap": 65_536,
        "simulator_search_lane_count": 2,
        "phase1_submission_environment": dict(common.PHASE1_SUBMISSION_ENVIRONMENT),
        "r240_hybrid_scheduler": dict(common.R242_BINDING_SCHEDULER),
        "deterministic_continuation": dict(common.BINDING_DETERMINISTIC_CONTINUATION),
    }
    proof = {
        "proof_kind": common.PROVEN_TERMINAL_WIN_PROOF_KIND,
        "root_observation_fingerprint": "sha256:current-root",
        "root_legal_order_fingerprint": "sha256:current-legal-order",
        "root_actor_seat": 0,
        "root_action": [1],
        "selected_action": [1],
        "terminal_result": "win",
        "terminal_winner_seat": 0,
        "terminal_leaf_reached": True,
        "proof_path_action_count": 1,
        "discovering_lane_id": 0,
        "path_actor_seats": [0],
        "path_no_chance_boundary": True,
        "path_no_actor_change_boundary": True,
        "path_no_opponent_boundary_crossing": True,
        "path_no_unresolved_randomness": True,
        "proof_is_deterministic": True,
    }
    receipt = common.passed_preflight_receipt(
        receipt_name=BUILDER.GATE_NAMES["terminal_win"],
        common_identity=identity,
        harness_schema=SMOKE.SCHEMA,
    )
    receipt.update(
        {
            "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
            "proven_deterministic_terminal_win_this_turn_regression_passed": True,
            "stop_reason": common.PROVEN_TERMINAL_WIN_STOP_REASON,
            "two_lane_topology_initialized_before_terminal_win_override": True,
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "search_begin_calls": 2,
            "search_release_calls": 2,
            "search_end_calls": 2,
            "completed_root_backup_count": 1,
            "terminal_win_proof_count": 1,
            "proven_deterministic_terminal_win_this_turn_stop_count": 1,
            "terminal_win_proof_backed_up_into_shared_root_tree": True,
            "terminal_leaf_returned_by_exact_stock_simulator": True,
            "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
            "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
            "outstanding_virtual_loss": 0,
            "two_independent_lane_proofs_required": False,
            "exhaustive_legal_action_scan_required": False,
            "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
            "terminal_win_proof": proof,
        }
    )
    BUILDER._receipt_common(
        receipt, receipt_name=BUILDER.GATE_NAMES["terminal_win"], identity=identity
    )
    projection = BUILDER._validate_r246_terminal_win_receipt(receipt, identity)
    assert projection["terminal_win_proof"] == proof
