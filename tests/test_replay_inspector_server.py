from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from replay_inspector.config import InspectorConfig
from replay_inspector.server import (
    InspectorApplication,
    InspectorHTTPError,
    _adapter_status,
    _factorized_action_transcript,
    _head_scales_query,
    _strict_int,
    create_server,
    main,
)


def _write_replay_archive(
    root: Path, *, submission_id: int = 77001, context: str = "Hand"
) -> tuple[int, int]:
    episode_id = 88001
    directory = root / str(submission_id)
    directory.mkdir(parents=True)
    observation = {
        "current": {
            "turn": 4,
            "yourIndex": 0,
            "players": [{"hand": []}, {"hand": None}],
        },
        "select": {
            "context": context,
            "option": [{"type": "Yes"}, {"type": "No"}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    replay = {
        "steps": [
            [
                {"status": "ACTIVE", "observation": observation},
                {"status": "INACTIVE", "observation": {}},
            ],
            [
                {"status": "ACTIVE", "observation": observation, "action": [1]},
                {"status": "INACTIVE", "observation": {}, "action": []},
            ],
        ]
    }
    (directory / f"episode-{episode_id}-replay.json").write_text(
        json.dumps(replay), encoding="utf-8"
    )
    (directory / "episodes.json").write_text(
        json.dumps(
            {
                "submission_id": submission_id,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "own_agent": {
                            "index": 0,
                            "team_name": "Elmo",
                            "reward": 2,
                        },
                        "opponent": "fixture-opponent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return submission_id, episode_id


def _write_two_seat_replay_archive(root: Path, *, own_seat: int) -> tuple[int, int]:
    """Write alternating seat decisions with an archive-declared own seat."""

    submission_id = 77001
    episode_id = 88001
    directory = root / str(submission_id)
    directory.mkdir(parents=True)

    def observation(*, turn: int, actor: int) -> dict[str, Any]:
        return {
            "current": {
                "turn": turn,
                "yourIndex": actor,
                "players": [{"hand": []}, {"hand": None}],
            },
            "select": {
                "context": "Hand",
                "option": [{"type": "Yes"}, {"type": "No"}],
                "minCount": 1,
                "maxCount": 1,
            },
        }

    def inactive(*, action: list[int] | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {"status": "INACTIVE", "observation": {}}
        if action is not None:
            row["action"] = action
        return row

    # Kaggle records the action for a step in the next seat row.  Seat 0 acts
    # at 0 and 4; seat 1 acts at 2.  Selecting one seat must not remove the
    # other seat's earlier transition from the replay passed to reconstruction.
    replay = {
        "steps": [
            [
                {"status": "ACTIVE", "observation": observation(turn=1, actor=0)},
                inactive(),
            ],
            [inactive(action=[1]), inactive()],
            [
                inactive(),
                {"status": "ACTIVE", "observation": observation(turn=2, actor=1)},
            ],
            [inactive(), inactive(action=[0])],
            [
                {"status": "ACTIVE", "observation": observation(turn=3, actor=0)},
                inactive(),
            ],
            [inactive(action=[0]), inactive()],
        ]
    }
    (directory / f"episode-{episode_id}-replay.json").write_text(
        json.dumps(replay), encoding="utf-8"
    )
    agents = [
        {
            "index": 0,
            "submission_id": submission_id if own_seat == 0 else 77002,
            "team_name": "Challengestone" if own_seat == 0 else "Fixture Opponent",
        },
        {
            "index": 1,
            "submission_id": submission_id if own_seat == 1 else 77002,
            "team_name": "Challengestone" if own_seat == 1 else "Fixture Opponent",
        },
    ]
    (directory / "episodes.json").write_text(
        json.dumps(
            {
                "submission_id": submission_id,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "own_agent": {
                            "index": own_seat,
                            "team_name": "Challengestone",
                            "submission_id": submission_id,
                        },
                        "agents": agents,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return submission_id, episode_id


def _config(tmp_path: Path) -> InspectorConfig:
    return InspectorConfig(
        replay_root=tmp_path / "archive",
        rollout_root=tmp_path / "rollouts",
        artifact_roots=(tmp_path / "artifacts",),
        web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
    )


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance_config(
    tmp_path: Path, *, include_runtime_parity_receipt: bool = True
) -> InspectorConfig:
    assets = tmp_path / "assets"
    assets.mkdir(parents=True)
    checkpoint = assets / "model.pt"
    bundle = assets / "submission.tar.gz"
    tree = assets / "matchup_tree.json"
    runtime_package = assets / "submitted-runtime.tar.gz"
    runtime_root = Path(__file__).resolve().parents[1] / "poke_bot"
    checkpoint.write_bytes(b"fixture checkpoint")
    bundle.write_bytes(b"fixture bundle")
    tree.write_text("{}", encoding="utf-8")
    runtime_package.write_bytes(b"fixture submitted runtime package")
    from replay_inspector.provenance import sha256_source_tree

    receipt = assets / "runtime-parity-receipt.json"
    if include_runtime_parity_receipt:
        receipt.write_text(
            json.dumps(
                {
                    "schema": "poke_bot.replay_model_inspector_runtime_parity_receipt/v1",
                    "version": 1,
                    "status": "verified",
                    "submission_id": 77001,
                    "checkpoint_sha256": _digest(checkpoint),
                    "bundle_sha256": _digest(bundle),
                    "runtime_package_sha256": _digest(runtime_package),
                    "runtime_source_tree_sha256": sha256_source_tree(runtime_root),
                    "verification": {
                        "method": "independent_exact_runtime_parity",
                        "verified_by": "test-fixture",
                        "verified_at_utc": "2026-08-07T00:00:00Z",
                    },
                }
            ),
            encoding="utf-8",
        )
    record: dict[str, Any] = {
        "submission_id": 77001,
        "status": "verified",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _digest(checkpoint),
        },
        "bundle": {"path": str(bundle), "sha256": _digest(bundle)},
        "matchup_tree": {"path": str(tree), "sha256": _digest(tree)},
        "runtime_package": {
            "path": str(runtime_package),
            "sha256": _digest(runtime_package),
        },
        "replay": {
            "games": [
                {
                    "episode_id": 88001,
                    "replay_sha256": _digest(
                        tmp_path / "archive" / "77001" / "episode-88001-replay.json"
                    ),
                }
            ]
        },
    }
    if include_runtime_parity_receipt:
        record["runtime_parity_receipt"] = {
            "path": str(receipt),
            "sha256": _digest(receipt),
        }
    manifest = tmp_path / "provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "poke_bot.replay_model_inspector_provenance/v1",
                "version": 1,
                "records": [record],
            }
        ),
        encoding="utf-8",
    )
    return InspectorConfig(
        replay_root=tmp_path / "archive",
        rollout_root=tmp_path / "rollouts",
        provenance_manifest=manifest,
        artifact_roots=(assets,),
        runtime_source_root=runtime_root,
        web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
    )


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], Any]:
    request = Request(f"{base_url}{path}", method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read()
            headers = dict(response.headers.items())
            return int(response.status), headers, json.loads(body.decode("utf-8"))
    except HTTPError as error:
        body = error.read()
        headers = dict(error.headers.items())
        return int(error.code), headers, json.loads(body.decode("utf-8"))


@pytest.fixture
def inspector_server(tmp_path: Path):
    submission_id, episode_id = _write_replay_archive(tmp_path / "archive")
    config = _config(tmp_path)
    application = InspectorApplication(config)
    server = create_server(config, application=application, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", submission_id, episode_id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_read_only_server_lists_replays_and_keeps_model_unavailable(
    inspector_server: tuple[str, int, int],
) -> None:
    base_url, submission_id, episode_id = inspector_server

    status, headers, health = _request(base_url, "/healthz")
    assert status == 200
    assert health["service"] == "replay-model-inspector"
    assert headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert "Access-Control-Allow-Origin" not in headers

    status, _headers, submissions = _request(base_url, "/api/submissions")
    assert status == 200
    assert submissions["submissions"][0]["submission_id"] == submission_id
    assert submissions["submissions"][0]["model_analysis"]["available"] is False
    assert submissions["submissions"][0]["label"] is None
    assert submissions["submissions"][0]["submission_label"]["available"] is False
    assert submissions["submissions"][0]["cached_replay_outcomes"] == {
        "available": True,
        "availability_reasons": [],
        "complete": True,
        "games_total": 1,
        "games_with_outcome": 1,
        "games_without_outcome": 0,
        "wins": 1,
        "nonwins": 0,
        "nonwins_definition": "source_backed_own_agent_reward_lte_zero",
        "draws": None,
        "draws_available": False,
        "draws_availability_reasons": [
            "draws_not_provable_from_own_agent_reward_semantics"
        ],
        "win_rate": 1.0,
        "source": "archived_episode_metadata.own_agent.reward",
    }

    status, _headers, games = _request(
        base_url, f"/api/submissions/{submission_id}/games"
    )
    assert status == 200
    assert games["submission_id"] == submission_id
    assert games["cached_replay_outcomes"]["win_rate"] == 1.0
    game = games["games"][0]
    assert game["episode_id"] == episode_id
    assert game["own_seat"] == 0
    assert game["opponent"] == "fixture-opponent"
    assert game["replay_available"] is True
    assert game["players"] == [
        {
            "seat": 0,
            "name": "Elmo",
            "name_available": True,
            "name_availability_reasons": [],
            "name_sources": ["archived_episode_metadata.own_agent.team_name"],
            "rank": None,
            "rank_available": False,
            "rank_availability_reasons": [
                "player_rank_unavailable_not_supplied_by_archived_metadata_or_replay"
            ],
            "rank_sources": [],
        },
        {
            "seat": 1,
            "name": None,
            "name_available": False,
            "name_availability_reasons": [
                "player_name_unavailable_not_supplied_by_archived_metadata_or_replay"
            ],
            "name_sources": [],
            "rank": None,
            "rank_available": False,
            "rank_availability_reasons": [
                "player_rank_unavailable_not_supplied_by_archived_metadata_or_replay"
            ],
            "rank_sources": [],
        },
    ]

    status, _headers, steps = _request(
        base_url, f"/api/submissions/{submission_id}/games/{episode_id}/steps"
    )
    assert status == 200
    assert steps["steps"] == [
        {
            "step_index": 0,
            "turn": 4,
            "context": "Hand",
            "candidate_count": 2,
            "recorded_action_summary": [1],
            "factorized_stages": [
                {
                    "stage": 0,
                    "candidate_count": 2,
                    "recorded_action": [1],
                    "recorded_stage_choice": [1],
                    "model_forward_expected": True,
                    "status": "decision",
                }
            ],
        }
    ]

    status, _headers, trace = _request(
        base_url,
        f"/api/submissions/{submission_id}/games/{episode_id}/steps/0?stage=0",
    )
    assert status == 200
    assert trace["availability"]["available"] is True
    assert trace["reproduction_status"] == "unavailable"
    assert trace["recorded_action"]["action"] == [1]
    assert trace["model"]["availability"]["available"] is False
    assert [option["action"] for option in trace["legal_options"]] == [[0], [1]]
    assert trace["legal_options"][0]["raw_action"] == [0]
    assert trace["legal_options"][0]["action_transcript"] == (
        "Factorized stage 0: Answer Yes."
    )
    assert trace["legal_options"][1]["action_transcript"] == (
        "Factorized stage 0: Answer No."
    )
    assert trace["legal_options"][1]["action_transcript_availability"] == {
        "available": True
    }
    assert trace["recorded_action"]["selected_action_raw"] == [1]
    assert trace["recorded_action"]["selected_action_transcript"] == (
        "Factorized stage 0: Answer No."
    )
    assert trace["model"]["adapter_status"]["status"] == "unavailable"

    status, _headers, parameters = _request(
        base_url, f"/api/submissions/{submission_id}/parameters"
    )
    assert status == 200
    assert parameters["availability"]["available"] is False
    assert parameters["parameters"] == []


def test_forced_setup_prompt_reports_exact_runtime_percentages(tmp_path: Path) -> None:
    submission_id, episode_id = _write_replay_archive(
        tmp_path / "archive", context="IsFirst"
    )
    application = InspectorApplication(_config(tmp_path))

    steps = application.steps_payload(submission_id, episode_id)
    assert steps["steps"][0]["factorized_stages"][0] == {
        "stage": 0,
        "candidate_count": 2,
        "recorded_action": [1],
        "recorded_stage_choice": [1],
        "model_forward_expected": False,
        "status": "forced_turn_order_or_runtime_short_circuit",
    }

    trace = application.trace_payload(submission_id, episode_id, 0, 0)
    assert trace["availability"]["available"] is True
    assert trace["reproduction_status"] == "exact_runtime_short_circuit"
    assert trace["model"]["availability"]["available"] is True
    assert trace["model"]["status"] == "deterministic_runtime_short_circuit"
    assert trace["model"]["neural_model_forward"] is False
    assert trace["model"]["selected_index"] == 1
    assert [option["probability"] for option in trace["legal_options"]] == [0.0, 1.0]
    assert [option["is_model_choice"] for option in trace["legal_options"]] == [
        False,
        True,
    ]
    assert trace["heads"] == []
    assert trace["decision_influence"]["availability"]["available"] is False
    assert (
        "no neural head influence"
        in trace["decision_influence"]["availability"]["reason"]
    )
    assert "100% / 0%" in trace["warnings"][0]


def test_static_security_and_get_only_routes(
    inspector_server: tuple[str, int, int],
) -> None:
    base_url, submission_id, _episode_id = inspector_server
    status, headers, _body = _request(base_url, "/app.js?cache=1")
    assert status == 400
    assert headers["X-Content-Type-Options"] == "nosniff"

    status, _headers, body = _request(
        base_url, "/api/submissions/77001/parameters/%2Fetc%2Fpasswd"
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_tensor_name"

    status, _headers, body = _request(base_url, "/api/submissions", method="POST")
    assert status == 405
    assert body["error"]["code"] == "method_not_allowed"

    status, _headers, body = _request(
        base_url,
        "/api/health",
        headers={"Origin": "https://example.invalid", "Sec-Fetch-Site": "cross-site"},
    )
    assert status == 403
    assert body["error"]["code"] == "cross_origin_request_rejected"

    status, _headers, body = _request(
        base_url, f"/api/submissions/{submission_id}/games/999/steps"
    )
    assert status == 404
    assert body["error"]["code"] == "episode_not_found"


@pytest.mark.parametrize(
    ("include_runtime_parity_receipt", "expected_status", "expected_dynamic_trace"),
    [
        (
            False,
            "weights_only",
            {"available": False, "reason": "runtime_parity_receipt_missing"},
        ),
        (True, "trace_ready", {"available": True}),
    ],
)
def test_submission_readiness_separates_weights_from_dynamic_trace(
    tmp_path: Path,
    include_runtime_parity_receipt: bool,
    expected_status: str,
    expected_dynamic_trace: dict[str, Any],
) -> None:
    _write_replay_archive(tmp_path / "archive")
    application = InspectorApplication(
        _provenance_config(
            tmp_path, include_runtime_parity_receipt=include_runtime_parity_receipt
        )
    )

    submission = application.submissions_payload()["submissions"][0]
    assert submission["status"] == expected_status
    assert submission["weights"] == {"available": True}
    assert submission["model_analysis"] == {"available": True}
    assert submission["dynamic_trace"] == expected_dynamic_trace

    games = application.games_payload(77001)
    assert games["status"] == expected_status
    assert games["weights"] == {"available": True}
    assert games["dynamic_trace"] == expected_dynamic_trace


@pytest.mark.parametrize(
    ("own_seat", "selectable_steps", "opponent_step", "history_opponent_step"),
    [
        (0, [0, 4], 2, 2),
        (1, [2], 0, 0),
    ],
)
def test_decision_selector_is_limited_to_archived_own_seat_and_keeps_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    own_seat: int,
    selectable_steps: list[int],
    opponent_step: int,
    history_opponent_step: int,
) -> None:
    submission_id, episode_id = _write_two_seat_replay_archive(
        tmp_path / "archive", own_seat=own_seat
    )
    application = InspectorApplication(_config(tmp_path))

    steps = application.steps_payload(submission_id, episode_id)
    assert steps["own_seat"] == own_seat
    assert steps["selectable_step_count"] == len(selectable_steps)
    assert [row["step_index"] for row in steps["steps"]] == selectable_steps
    assert steps["decision_owner"] == {
        "availability": {"available": True},
        "seat": own_seat,
        "seat_authority": "archived_submission_bound_own_agent_seat",
        "selection_scope": "own_agent_decisions_only",
        "identity": {
            "available": True,
            "name": "Challengestone",
            "source": "archived_episode_metadata.own_agent.team_name",
        },
    }

    captured: dict[str, Any] = {}

    def fake_trace(context: Any, *, head_scales: Any = None) -> dict[str, Any]:
        captured["replay"] = context.replay
        captured["seat"] = context.seat
        return {
            "model": {
                "availability": {"available": True},
                "status": "available",
                "selected_index": 0,
            },
            "heads": [],
            "fusion": None,
            "provenance": None,
            "reproduction_status": "recomputed_not_historical",
            "warnings": [],
        }

    monkeypatch.setattr(application, "_trace_analysis_reason", lambda *_args: None)
    monkeypatch.setattr(application, "_inspect_exact_trace", fake_trace)
    trace = application.trace_payload(
        submission_id, episode_id, selectable_steps[-1], 0
    )
    assert captured["seat"] == own_seat
    assert len(captured["replay"]["steps"]) == 6
    # The other player's prior turn survives untouched in the causal replay;
    # only its URL is excluded from model attribution/selection.
    other_seat = 1 - own_seat
    assert (
        captured["replay"]["steps"][history_opponent_step][other_seat]["status"]
        == "ACTIVE"
    )
    assert captured["replay"]["steps"][history_opponent_step + 1][other_seat][
        "action"
    ] in ([0], [1])
    assert trace["address"]["own_seat"] == own_seat
    assert trace["address"]["decision_owner"]["identity"]["name"] == "Challengestone"

    captured.clear()
    with pytest.raises(InspectorHTTPError) as error:
        application.trace_payload(submission_id, episode_id, opponent_step, 0)
    assert error.value.status == 404
    assert error.value.code == "opponent_decision_not_selectable"
    assert captured == {}


@pytest.mark.parametrize(
    "own_agent, extra_episode_fields, expected_reason",
    [
        ({"team_name": "Challengestone"}, {}, "own_seat_not_declared"),
        (
            {"index": 0, "team_name": "Challengestone"},
            {"own_seat": 1},
            "own_seat_ambiguous",
        ),
        (
            {
                "index": 0,
                "team_name": "Challengestone",
                "submission_id": 77002,
            },
            {},
            "own_agent_submission_id_mismatch",
        ),
    ],
)
def test_decision_selector_fails_closed_without_one_archived_own_seat(
    tmp_path: Path,
    own_agent: dict[str, Any],
    extra_episode_fields: dict[str, Any],
    expected_reason: str,
) -> None:
    submission_id, episode_id = _write_replay_archive(tmp_path / "archive")
    episodes_path = tmp_path / "archive" / str(submission_id) / "episodes.json"
    document = json.loads(episodes_path.read_text(encoding="utf-8"))
    episode = document["episodes"][0]
    episode["own_agent"] = own_agent
    episode.update(extra_episode_fields)
    episodes_path.write_text(json.dumps(document), encoding="utf-8")
    application = InspectorApplication(_config(tmp_path))

    with pytest.raises(InspectorHTTPError) as error:
        application.steps_payload(submission_id, episode_id)
    assert error.value.status == 422
    assert error.value.code == "own_seat_unavailable"
    assert error.value.details == {"reason": expected_reason}


@pytest.mark.parametrize(
    ("own_seat", "actor", "missing"),
    [
        (0, None, True),
        (0, 1, False),
        (1, 0, False),
        (0, True, False),
        (1, "1", False),
    ],
)
def test_active_select_requires_exact_archived_actor_identity(
    tmp_path: Path,
    own_seat: int,
    actor: object,
    missing: bool,
) -> None:
    submission_id, episode_id = _write_two_seat_replay_archive(
        tmp_path / "archive", own_seat=own_seat
    )
    replay_path = (
        tmp_path / "archive" / str(submission_id) / f"episode-{episode_id}-replay.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    active_step = 0 if own_seat == 0 else 2
    current = replay["steps"][active_step][own_seat]["observation"]["current"]
    if missing:
        del current["yourIndex"]
    else:
        current["yourIndex"] = actor
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    application = InspectorApplication(_config(tmp_path))

    with pytest.raises(InspectorHTTPError) as steps_error:
        application.steps_payload(submission_id, episode_id)
    assert steps_error.value.status == 422
    assert steps_error.value.code == "replay_timeline_unavailable"

    with pytest.raises(InspectorHTTPError) as trace_error:
        application.trace_payload(submission_id, episode_id, active_step, 0)
    assert trace_error.value.status == 422
    assert trace_error.value.code == "replay_timeline_unavailable"


def test_submission_id_text_preserves_exact_decimal_above_browser_safe_integer(
    tmp_path: Path,
) -> None:
    submission_id = 9_007_199_254_740_993
    expected_text = "9007199254740993"
    _submission_id, episode_id = _write_replay_archive(
        tmp_path / "archive", submission_id=submission_id
    )
    application = InspectorApplication(_config(tmp_path))

    submission = application.submissions_payload()["submissions"][0]
    assert submission["submission_id"] == submission_id
    assert submission["submission_id_text"] == expected_text
    assert "e" not in submission["submission_id_text"].casefold()

    games = application.games_payload(submission_id)
    steps = application.steps_payload(submission_id, episode_id)
    trace = application.trace_payload(submission_id, episode_id, 0, 0)
    parameters = application.parameters_payload(submission_id)
    parameter = application.parameter_detail_payload(
        submission_id, "missing", offset=0, limit=1, bins=1
    )
    for payload in (games, steps, trace, parameters, parameter):
        assert payload["submission_id"] == submission_id
        assert payload["submission_id_text"] == expected_text
    assert games["games"][0]["submission_id_text"] == expected_text
    assert trace["address"]["submission_id_text"] == expected_text

    assert _strict_int(expected_text, field="submission_id", minimum=1) == submission_id
    for invalid in ("9.007199254740993e15", "9007199254740993e0", "+9007199254740993"):
        with pytest.raises(InspectorHTTPError, match="decimal integer"):
            _strict_int(invalid, field="submission_id", minimum=1)


def test_factorized_transcripts_distinguish_synthetic_stop_from_engine_end() -> None:
    observation = {
        "select": {
            "option": [
                {"type": "End"},
                {"type": "Attack", "attackId": 17},
                {"type": "FutureUnsupportedOption"},
            ]
        }
    }
    stop = _factorized_action_transcript(
        observation, [], factorized_stage=0, recorded_action=[]
    )
    engine_end = _factorized_action_transcript(
        observation, [0], factorized_stage=0, recorded_action=[]
    )
    unknown = _factorized_action_transcript(
        observation, [2], factorized_stage=0, recorded_action=[]
    )

    assert stop["synthetic_stop"] is True
    assert "synthetic STOP" in stop["text"]
    assert engine_end["synthetic_stop"] is False
    assert "engine's End option" in engine_end["text"]
    assert unknown["availability"]["available"] is False
    assert "raw option path" in unknown["text"]


def test_adapter_status_requires_actual_routed_runtime_use() -> None:
    activation = {
        "status": "applied",
        "applied": True,
        "evaluation_basis": "checksum_bound_causal_re_evaluation",
        "historical_activation_recorded": False,
    }
    active = _adapter_status(
        {
            "bank_present": True,
            "runtime_enabled": True,
            "decision_route_active": True,
            "route": 4,
            "slot": 4,
            "route_routable": True,
            "matched_archetype": "known-matchup",
            "route_reliability": {"availability": {"available": True}},
            "policy_influence": {"availability": {"available": True}},
            "runtime_activation": activation,
        }
    )
    bypassed = _adapter_status(
        {
            "bank_present": True,
            "runtime_enabled": True,
            "decision_route_active": False,
            "route": -1,
            "route_routable": None,
            "runtime_activation": activation,
        }
    )
    installed_but_unverified = _adapter_status(
        {
            "bank_present": True,
            "runtime_enabled": True,
            "decision_route_active": False,
            "route": 4,
            "route_routable": True,
            "runtime_activation": activation,
        }
    )
    no_tree = _adapter_status(
        {
            "bank_present": True,
            "runtime_enabled": False,
            "decision_route_active": False,
            "route": None,
            "route_routable": None,
            "runtime_activation": {
                "status": "unavailable",
                "applied": False,
                "reason": (
                    "checksum-bound submitted startup activation evidence was not supplied"
                ),
            },
        }
    )

    assert active["status"] == "active_for_decision"
    assert bypassed["status"] == "bypassed"
    assert "unknown/bypass route" in str(bypassed["reason"])
    assert installed_but_unverified["status"] == "unavailable"
    assert "cannot be verified" in str(installed_but_unverified["reason"])
    assert no_tree["status"] == "unavailable"
    assert "startup activation evidence" in str(no_tree["reason"])


def test_head_scales_get_query_is_bounded_and_unambiguous() -> None:
    assert _head_scales_query(None) is None
    assert _head_scales_query("value:0,action_type:1.5,combo_state:2") == {
        "value": 0.0,
        "action_type": 1.5,
        "combo_state": 2.0,
    }
    with pytest.raises(Exception, match="only once"):
        _head_scales_query("value:1,value:2")
    with pytest.raises(Exception, match="between 0 and 2"):
        _head_scales_query("value:2.01")
    with pytest.raises(Exception, match="malformed"):
        _head_scales_query("Value:1")


def test_loopback_guard_and_check_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    _write_replay_archive(config.replay_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "bind_host": "127.0.0.1",
                "port": 8791,
                "replay_root": str(config.replay_root),
                "rollout_root": str(config.rollout_root),
                "artifact_roots": [str(path) for path in config.artifact_roots],
                "web_root": str(config.web_root),
            }
        ),
        encoding="utf-8",
    )
    assert main(config_path=config_path, check=True) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["check"]["static_assets_available"] is True
    with pytest.raises(ValueError, match="loopback-only"):
        create_server(config, host="0.0.0.0", port=8791)


def test_trace_normalizes_checksum_bound_inference_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_replay_archive(tmp_path / "archive")
    application = InspectorApplication(_provenance_config(tmp_path))

    from poke_bot import public_matchup_router

    monkeypatch.setattr(
        public_matchup_router.PublicMatchupDecisionTree,
        "from_path",
        classmethod(
            lambda _cls, path, **_kwargs: SimpleNamespace(digest=_digest(Path(path)))
        ),
    )

    class FakeModel:
        def named_parameters(self):
            return []

    class FakeCache:
        def load(self, checkpoint_path: Path, expected_digest: str):
            assert checkpoint_path.name == "model.pt"
            assert expected_digest.startswith("sha256:")
            return SimpleNamespace(model=FakeModel())

    captured: dict[str, Any] = {}

    def fake_inspect(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "availability": {"available": True},
            "provenance": {
                **kwargs["provenance"],
                "reproduction_status": "recomputed_not_historical",
            },
            "replay": {"legal_candidates": [[0], [1]]},
            "policy": {
                "base_logits": [0.1, 0.2],
                "final_logits": [0.3, 0.4],
                "probabilities": [0.45, 0.55],
                "model_choice_index": 1,
                "model_choice_candidate": [1],
                "model_matches_recorded_target": True,
            },
            "value": {"policy_value": [0.75]},
            "adapter": {
                "bank_present": True,
                "runtime_enabled": True,
                "decision_route_active": True,
                "route": 2,
                "slot": 2,
                "route_routable": True,
                "matched_archetype": "fixture-matchup",
                "matched_archetype_source": "router_audit",
                "route_reliability": {
                    "availability": {"available": True},
                    "kind": "fixed_residual_scale",
                    "residual_scale": 0.1,
                },
                "policy_influence": {
                    "availability": {"available": True},
                    "method": "full_final_policy_minus_policy_without_matchup_adapter_route",
                    "sign_convention": "full_policy_minus_policy_without_matchup_adapter_route",
                    "selected_option_index": 1,
                    "selected_option_probability_delta": 0.05,
                    "selected_option_logit_delta": 0.2,
                    "maximum_absolute_option_probability_delta": 0.05,
                    "total_variation_distance": 0.05,
                    "most_helped_option": {"index": 1, "probability_delta": 0.05},
                    "most_hurt_option": {"index": 0, "probability_delta": -0.05},
                    "leave_one_adapter_out": {
                        "full_policy_logits": [0.3, 0.4],
                        "policy_without_head_logits": [0.5, 0.1],
                        "full_policy_probabilities": [0.45, 0.55],
                        "policy_without_head_probabilities": [0.6, 0.4],
                        "effect_logits": [-0.2, 0.3],
                        "effect_probabilities": [-0.15, 0.15],
                        "ablated_model_choice_index": 0,
                    },
                },
                "runtime_activation": {
                    "status": "applied",
                    "applied": True,
                    "evaluation_basis": "checksum_bound_causal_re_evaluation",
                    "historical_activation_recorded": False,
                    "cached_model_state_restored_after_request": True,
                },
            },
            "heads": {
                "value": {
                    "availability": {"available": True},
                    "name": "value",
                    "source_kind": "state",
                    "raw_values": [0.75],
                    "normalization": {"kind": "tanh", "values": [0.64]},
                    "mask": {"legal_candidate_mask": None},
                    "policy_influence": {
                        "availability": {"available": True},
                        "method": "exact_leave_one_head_out_final_policy_recomputation",
                        "sign_convention": "full_policy_minus_policy_without_head",
                        "selected_option_index": 1,
                        "selected_option_probability_delta": 0.03,
                        "selected_option_logit_delta": 0.1,
                        "maximum_absolute_option_probability_delta": 0.03,
                        "total_variation_distance": 0.03,
                        "most_helped_option": {"index": 1, "probability_delta": 0.03},
                        "most_hurt_option": {"index": 0, "probability_delta": -0.03},
                    },
                }
            },
            "fusion": {
                "availability": {"available": True},
                "routes": {
                    "routes": {
                        "value": {
                            "availability": {"available": True},
                            "runtime_active": True,
                            "reliability": {"effective_multiplier": 1.2},
                            "raw_route_delta": [0.01, 0.02],
                            "runtime_contribution": [0.012, 0.024],
                        }
                    }
                },
                "leave_one_out": {"value": {"changes_model_choice": False}},
            },
            "latent_lookahead": {"availability": {"available": False}},
            "decision_influence": {
                "availability": {"available": True},
                "schema": "poke_bot.replay_model_inspector.decision_influence/v1",
                "mode": "decision_only_counterfactual",
                "method": "exact_nonlinear_fusion_source_recomputation",
                "training_weight": False,
                "historical": False,
                "reproduction_status": "recomputed_not_historical",
                "default_scale": 1.0,
                "scale_bounds": [0.0, 2.0],
                "requested_scales": {"value": 1.5},
                "effective_scales": {"value": 1.5},
                "eligible_heads": ["value"],
                "baseline": {
                    "final_logits": [0.3, 0.4],
                    "probabilities": [0.45, 0.55],
                    "selected_option_index": 1,
                },
                "counterfactual": {
                    "final_logits": [0.25, 0.45],
                    "probabilities": [0.4, 0.6],
                    "selected_option_index": 1,
                },
                "effect": {
                    "most_helped_option": {
                        "index": 1,
                        "probability_delta": 0.05,
                    },
                    "most_hurt_option": {
                        "index": 0,
                        "probability_delta": -0.05,
                    },
                },
            },
            "guide_shadow": {
                "schema": "poke_bot.replay_model_inspector.guide_shadow/v1",
                "availability": {"available": True},
                "guide_id": "fixture-guide",
                "recommended_index": 1,
                "policy_authority": False,
                "policy_logit_delta": 0.0,
            },
        }

    application._model_cache = FakeCache()  # type: ignore[assignment]
    application._inference_module = SimpleNamespace(inspect_replay_step=fake_inspect)
    trace = application.trace_payload(77001, 88001, 0, 0, head_scales={"value": 1.5})

    assert captured["acting_seat"] == 0
    assert captured["env_step"] == 0
    assert captured["factorized_stage"] == 0
    assert captured["provenance"]["runtime_parity_verified"] is True
    activation = captured["submitted_runtime_activation"]
    assert activation["basis"] == "checksum_bound_submitted_startup"
    assert activation["matchup_tree_verified"] is True
    assert activation["submitted_startup_behavior_verified"] is True
    assert captured["head_scales"] == {"value": 1.5}
    assert trace["reproduction_status"] == "recomputed_not_historical"
    assert trace["model"]["selected_index"] == 1
    assert trace["model"]["value"] == 0.75
    assert trace["guide_shadow"]["guide_id"] == "fixture-guide"
    assert trace["guide_shadow"]["policy_authority"] is False
    assert trace["legal_options"][1]["probability"] == 0.55
    assert (
        trace["legal_options"][1]["route_contributions"]["value"]["contribution"]
        == 0.024
    )
    head = trace["heads"][0]
    assert head["name"] == "value"
    assert head["raw"] == [0.75]
    assert head["route_delta"] == [0.012, 0.024]
    influence = head["policy_influence"]
    assert influence["availability"] == {"available": True}
    assert influence["selected_option"]["label"] == ("Factorized stage 0: Answer No.")
    assert influence["most_hurt_option"]["label"] == ("Factorized stage 0: Answer Yes.")
    assert "non-additive" in influence["plain_english_interpretation"]
    adapter_status = trace["model"]["adapter_status"]
    assert adapter_status["status"] == "active_for_decision"
    assert adapter_status["enabled"] is True
    assert adapter_status["matched_archetype"] == "fixture-matchup"
    assert adapter_status["slot"] == 2
    assert adapter_status["policy_influence"]["selected_option"]["label"] == (
        "Factorized stage 0: Answer No."
    )
    assert (
        "Matchup Adapter route"
        in adapter_status["policy_influence"]["plain_english_interpretation"]
    )
    comparison = adapter_status["on_off_comparison"]
    assert comparison["availability"] == {"available": True}
    assert comparison["adapter_on"]["selected_option"]["label"] == (
        "Factorized stage 0: Answer No."
    )
    assert comparison["adapter_off"]["selected_option"]["label"] == (
        "Factorized stage 0: Answer Yes."
    )
    assert comparison["choice_changed"] is True
    assert comparison["options"][1]["probability_delta"] == pytest.approx(0.15)
    assert comparison["sign_convention"] == "adapter_on_minus_adapter_off"
    decision_influence = trace["decision_influence"]
    assert decision_influence["training_weight"] is False
    assert decision_influence["baseline"]["selected_option"]["label"] == (
        "Factorized stage 0: Answer No."
    )
    assert decision_influence["effect"]["most_hurt_option"]["label"] == (
        "Factorized stage 0: Answer Yes."
    )


def test_trace_fails_closed_without_configured_extracted_runtime(
    tmp_path: Path,
) -> None:
    _write_replay_archive(tmp_path / "archive")
    config = replace(_provenance_config(tmp_path), runtime_source_root=None)
    application = InspectorApplication(config)

    class NeverLoadCache:
        def load(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("dynamic trace must be rejected before model load")

    application._model_cache = NeverLoadCache()  # type: ignore[assignment]
    trace = application.trace_payload(77001, 88001, 0, 0)

    assert trace["model"]["availability"] == {
        "available": False,
        "reason": "runtime_source_tree_sha256_mismatch",
    }
    assert trace["heads"] == []


def test_trace_rehashes_matchup_tree_before_loading_cached_model(
    tmp_path: Path,
) -> None:
    _write_replay_archive(tmp_path / "archive")
    config = _provenance_config(tmp_path)
    application = InspectorApplication(config)
    tree = tmp_path / "assets" / "matchup_tree.json"
    tree.write_text('{"changed":true}', encoding="utf-8")

    class NeverLoadCache:
        def load(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("tree mismatch must fail before model load")

    application._model_cache = NeverLoadCache()  # type: ignore[assignment]
    trace = application.trace_payload(77001, 88001, 0, 0)

    assert trace["model"]["availability"] == {
        "available": False,
        "reason": "matchup_tree_sha256_mismatch",
    }
    assert trace["heads"] == []
