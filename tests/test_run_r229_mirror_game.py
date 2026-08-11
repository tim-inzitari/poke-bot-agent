from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_r229_mirror_game.py"
spec = importlib.util.spec_from_file_location("r229_game", SCRIPT)
assert spec and spec.loader
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


def _package(monkeypatch, tmp_path: Path) -> Path:
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    libraries = {}
    for platform_name, (relative, _digest, _size) in game.CANONICAL_NATIVE_LIBRARIES.items():
        payload = (platform_name + "\n").encode()
        path = package / relative
        path.write_bytes(payload)
        libraries[platform_name] = (
            relative,
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
    monkeypatch.setattr(game, "CANONICAL_NATIVE_LIBRARIES", libraries)
    (package / "r244_fleet_evaluation_manifest.json").write_text(json.dumps({
        "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r244_package/v1",
        "owner_goal_revision": 244,
        "canonical_libcg_revision": 236,
        "owner_two_lane_topology_revision": 239,
        "owner_handle_scoped_search_id_revision": 244,
        "simulator_lane_count": 2,
        "internal_agent_start_arena_count": 2,
        "required_search_begin_call_count": 2,
        "required_distinct_per_lane_handle_identity_count": 2,
        "required_handle_scoped_search_id_chain_count": 2,
        "required_distinct_handle_first_search_id_composite_count": 2,
        "handle_scoped_first_search_id_composite_state_array_field": (
            "handle_scoped_first_search_id_composite_states"
        ),
        "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order": [
            "lane_id",
            "handle_identity",
            "first_search_id",
        ],
        "search_begin_identity_scope": "arena_handle_plus_handle_local_search_id",
        "raw_search_id_global_uniqueness_required": False,
        "logical_frontier_leaf_count_per_frozen_model_batch": 2,
        "partial_frontier_batches_allowed": False,
        "serial_one_lane_continuation_allowed": False,
        "one_shared_logical_mcts_tree_required": True,
        "bo_lifecycle_revision": 233,
        "r234_kaggle_broker_or_queue_lifecycle_included": False,
        "canonical_native_libraries": {
            name: {"path": relative, "sha256": digest, "size_bytes": size}
            for name, (relative, digest, size) in libraries.items()
        },
    }))
    return package


def test_game_preflight_binds_complete_set_and_host_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    monkeypatch.setattr(game.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(game.platform, "machine", lambda: "arm64")
    complete, selected = game._verify_canonical_native_set(package)
    assert set(complete) == set(game.CANONICAL_NATIVE_LIBRARIES)
    assert selected["platform_identity"] == "macos_arm64"
    assert selected["path"] == "cg/libcg.dylib"


def test_game_preflight_rejects_mixed_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="drifted"):
        game._verify_canonical_native_set(package)


def test_game_preflight_rejects_extra_native_member(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    (package / "cg/libcg-old.so").write_bytes(b"historical")
    with pytest.raises(game.R229GameError, match="mixed or incomplete"):
        game._verify_canonical_native_set(package)


def test_game_preflight_rejects_eight_lane_manifest(monkeypatch, tmp_path):
    package = _package(monkeypatch, tmp_path)
    manifest = package / "r244_fleet_evaluation_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["simulator_lane_count"] = 8
    manifest.write_text(json.dumps(payload))
    with pytest.raises(game.R229GameError, match="manifest identity drifted"):
        game._verify_canonical_native_set(package)


def _two_lane_receipt():
    return {
        "mode": "shared_tree_mcts",
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "per_lane_handle_identities": [1001, 1002],
        "search_begin_calls": 2,
        "search_release_calls": 2,
        "search_end_calls": 2,
        "max_simulator_calls_in_flight": 2,
        "per_lane_depth": [1, 1],
        "per_lane_search_id_chains": [[0], [0]],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": 1001, "first_search_id": 0},
            {"lane_id": 1, "handle_identity": 1002, "first_search_id": 0},
        ],
        "microbatch_sizes": [2],
        "outstanding_virtual_loss": 0,
    }


def test_decision_receipt_requires_two_distinct_lanes_and_full_batches():
    game._validate_two_lane_receipts([_two_lane_receipt()])
    bad = _two_lane_receipt()
    bad["per_lane_handle_identities"] = [1001, 1001]
    with pytest.raises(game.R229GameError, match="exact two-lane"):
        game._validate_two_lane_receipts([bad])
    bad = _two_lane_receipt()
    bad["handle_scoped_first_search_id_composite_states"][1]["handle_identity"] = 1001
    with pytest.raises(game.R229GameError, match="exact two-lane"):
        game._validate_two_lane_receipts([bad])
    bad = _two_lane_receipt()
    bad["handle_scoped_first_search_id_composite_states"][0] = {
        "handle_identity": 1001,
        "lane_id": 0,
        "first_search_id": 0,
    }
    with pytest.raises(game.R229GameError, match="exact two-lane"):
        game._validate_two_lane_receipts([bad])
    bad = _two_lane_receipt()
    bad["microbatch_sizes"] = [1]
    with pytest.raises(game.R229GameError, match="exact two-lane"):
        game._validate_two_lane_receipts([bad])
