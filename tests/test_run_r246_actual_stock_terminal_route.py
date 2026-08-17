"""CPU-only fail-closed coverage for the r246 actual-stock terminal worker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_r246_actual_stock_terminal_route.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("r246_actual_terminal_worker_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_module()


def _source_payload(*, terminal_reward: int = 1) -> dict[str, object]:
    """Small JSON-native shape of the exact reviewed physical root."""

    root = {
        "current": {
            "result": -1,
            "yourIndex": 1,
            "players": [{"deckCount": 0, "hand": None}, {"deckCount": 7}],
        },
        "search_begin_input": "captured-stock-search-root",
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 7}] * 6 + [{"type": 14}],
        },
    }
    steps: list[list[dict[str, object]]] = [[{}, {}] for _ in range(225)]
    steps[223][1] = {"observation": root, "action": [6]}
    steps[224][0] = {
        "observation": {"current": {"result": -1}},
        "action": [],
        "reward": -1,
        "status": "DONE",
    }
    steps[224][1] = {
        "observation": {"current": {"result": -1}},
        "action": [6],
        "reward": terminal_reward,
        "status": "DONE",
    }
    return {"steps": steps}


def _write_source(tmp_path: Path, *, terminal_reward: int = 1) -> Path:
    path = tmp_path / "episode-89740321-replay.json"
    path.write_text(json.dumps(_source_payload(terminal_reward=terminal_reward)), encoding="utf-8")
    return path


def _marker() -> dict[str, object]:
    """A literal-shaped staged marker; validator truth is separately mocked."""

    return {
        "mode": "shared_tree_mcts",
        "stop_reason": WORKER.R246_STOP_REASON,
        "selected_action": [6],
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "mcts_action_authority": True,
        "degraded": False,
        "direct_action_precomputed_and_validated": True,
        "broker_started": True,
        "mcts_child_started": True,
        "mcts_child_called": True,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        "completed_root_backup_count": 1,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "child_search_elapsed_seconds": 0.01,
        "parent_action_elapsed_seconds": 0.02,
    }


def _validated_terminal_proof() -> dict[str, object]:
    return {
        "proof_kind": WORKER.R246_PROOF_KIND,
        "root_action": [6],
        "selected_action": [6],
    }


def _marker_text(marker: dict[str, object]) -> str:
    return "R238_TWO_LANE_BOUNDED_MCTS_DECISION " + json.dumps(marker) + "\n"


def test_archived_root_requires_exact_physical_shape_and_recorded_terminal_successor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _write_source(tmp_path)
    monkeypatch.setattr(WORKER, "sha256_file", lambda _path: WORKER.EXPECTED_REPLAY_SHA256)

    candidate = WORKER._read_archived_physical_root(source)

    assert candidate.observation["current"]["yourIndex"] == 1
    assert candidate.source["recorded_root_action"] == [6]
    assert candidate.source["recorded_episode_reward_winner_seat"] == 1
    assert candidate.source["stage_deck_reachability_claimed"] is False


def test_archived_root_rejects_nonwinning_successor_even_when_sha_is_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _write_source(tmp_path, terminal_reward=0)
    monkeypatch.setattr(WORKER, "sha256_file", lambda _path: WORKER.EXPECTED_REPLAY_SHA256)

    with pytest.raises(WORKER.ActualStockTerminalRouteError, match="not a recorded win"):
        WORKER._read_archived_physical_root(source)


def test_literal_marker_is_not_accepted_without_every_outer_r246_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = _marker()
    marker.pop("broker_started")
    monkeypatch.setattr(
        WORKER,
        "validate_decision_marker",
        lambda *_args, **_kwargs: {"terminal_win_proof": _validated_terminal_proof()},
    )

    with pytest.raises(WORKER.ActualStockTerminalRouteError, match="broker_started"):
        WORKER._require_literal_r246_marker(
            staged_stdout=_marker_text(marker),
            action=[6],
            legal_actions=WORKER.EXPECTED_LEGAL_ORDER,
            observation={"current": {"yourIndex": 1}},
        )


def test_literal_marker_requires_ambiguous_mcts_not_a_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = _marker()
    marker["mode"] = "high_confidence_frozen_direct"
    monkeypatch.setattr(
        WORKER,
        "validate_decision_marker",
        lambda *_args, **_kwargs: {"terminal_win_proof": _validated_terminal_proof()},
    )

    with pytest.raises(WORKER.ActualStockTerminalRouteError, match="ambiguous MCTS"):
        WORKER._require_literal_r246_marker(
            staged_stdout=_marker_text(marker),
            action=[6],
            legal_actions=WORKER.EXPECTED_LEGAL_ORDER,
            observation={"current": {"yourIndex": 1}},
        )


def test_run_emits_only_a_verbatim_staged_terminal_marker_after_one_agent_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = tmp_path / "candidate.tar.gz"
    manifest = tmp_path / "manifest.json"
    r225 = tmp_path / "r225.json"
    r236 = tmp_path / "r236.json"
    for path in (archive, manifest, r225, r236):
        path.write_text("{}", encoding="utf-8")

    root = WORKER.ArchivedPhysicalRoot(
        observation={"current": {"yourIndex": 1}, "select": {"option": []}},
        source={"kind": "test_archived_physical_root"},
    )
    marker = _marker()
    calls: list[dict[str, object]] = []

    class FakeBroker:
        closed = False

        def close(self) -> None:
            self.closed = True

    broker = FakeBroker()

    class FakeMain:
        _BROKER = broker

        @staticmethod
        def agent(observation: dict[str, object]) -> list[int]:
            calls.append(observation)
            print(_marker_text(marker), end="")
            return [6]

    class FakeFeatures:
        @staticmethod
        def enumerate_action_combos(
            _observation: dict[str, object], *, max_combos: int
        ) -> list[list[int]]:
            assert max_combos == WORKER.COMPLETE_ACTION_CAP
            return [list(action) for action in WORKER.EXPECTED_LEGAL_ORDER]

    identity = {
        "common_identity": {"common": "exact"},
        "exact_package": {"package": "exact"},
        "stage_contract": {"contract": "exact"},
    }
    monkeypatch.setattr(WORKER, "_read_archived_physical_root", lambda _path: root)
    monkeypatch.setattr(WORKER, "load_binding_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        WORKER,
        "stage_snapshot",
        lambda _stage: {"tree_sha256": "sha256:unchanged"},
    )
    monkeypatch.setattr(WORKER, "_load_exact_stage", lambda _stage: (FakeMain, FakeFeatures))
    monkeypatch.setattr(
        WORKER,
        "validate_decision_marker",
        lambda *_args, **_kwargs: {"terminal_win_proof": _validated_terminal_proof()},
    )

    result = WORKER._run(
        stage=stage,
        candidate_archive=archive,
        member_manifest=manifest,
        r225_contract=r225,
        r236_contract=r236,
        source_replay=tmp_path / "ignored-by-mock.json",
    )

    witness = result["r240_witnesses"][
        "synthetic_proven_deterministic_terminal_win_this_turn"
    ]
    assert witness == marker
    assert result["witness_origin"] == "actual_stock_search_route"
    assert result["sealed_parent"]["agent_call_count"] == 1
    assert len(calls) == 1
    assert broker.closed is True
    assert result["stage_mutation_check"]["unchanged"] is True


def test_real_archived_candidate_digest_is_checked_when_local_evidence_exists() -> None:
    """Keep the durable candidate path auditable without requiring ignored evidence in CI."""

    source = (
        ROOT
        / "outputs/analysis/kaggle-marnie-iter0-4-20260803/iter_00004/"
        "episode-89740321-replay.json"
    )
    if not source.is_file():
        pytest.skip("local archived replay evidence is intentionally not version-controlled")
    assert WORKER.sha256_file(source) == WORKER.EXPECTED_REPLAY_SHA256
    candidate = WORKER._read_archived_physical_root(source)
    assert candidate.observation["current"]["yourIndex"] == WORKER.EXPECTED_SEAT
    assert candidate.source["recorded_episode_reward_winner_seat"] == WORKER.EXPECTED_SEAT
