"""Focused integrity coverage for the r212 raw-public route sidecars.

These tests deliberately use tiny synthetic raw episodes and feature shards.
They exercise the production scanner and immutable sidecar reader without
depending on a protected multi-gigabyte archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any, ClassVar

import pytest

from poke_bot import features
from poke_bot import guide2vec_public_routes as routes
from poke_bot.dataset import DecisionSample, GameSequence
from poke_bot.feature_shards import (
    DATASET_CACHE_SCHEMA_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from scripts import materialize_alakazam_guide2vec_r212_public_routes as materializer

DAY = "2026-07-22"
DAY_TWO = "2026-07-23"
TREE_SHA256 = "sha256:" + "1" * 64
BUNDLE_SHA256 = "sha256:" + "2" * 64
ENTRYPOINT_SHA256 = "sha256:" + "3" * 64
ROUTER_SHA256 = "sha256:" + "4" * 64
RAW_SHA256 = "sha256:" + "5" * 64
OTHER_RAW_SHA256 = "sha256:" + "6" * 64
RAW_MEMBER_SHA256 = "sha256:" + "7" * 64
PRODUCER_ROUTE_SHA256 = "sha256:" + "8" * 64
PRODUCER_CLI_SHA256 = "sha256:" + "9" * 64


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse(feature: int) -> features.SparseVector:
    vector = features.SparseVector()
    vector.word_start()
    vector.add(feature, 1.0)
    vector.add_pos(feature + 2)
    return vector


def _sequence(*, episode_id: str = "episode-route", env_step: int = 0) -> GameSequence:
    decision = DecisionSample(
        board=_sparse(3),
        options=_sparse(4),
        action=[7],
        action_combo_index=0,
        action_combos=[[7]],
        env_step=env_step,
        action_token=_sparse(7),
    )
    return GameSequence(
        episode_id=episode_id,
        seat=0,
        archetype="alakazam",
        opp_archetype="unknown",
        deck=[1] * 60,
        value=0.0,
        decisions=[decision],
    )


def _alignment(sequence: GameSequence) -> str:
    return routes.compact_alignment_sha256(
        env_steps=[decision.env_step for decision in sequence.decisions],
        boards=[decision.board for decision in sequence.decisions],
        action_tokens=[decision.action_token for decision in sequence.decisions],
    )


def _raw_archive(*, day: str = DAY, sha256: str = RAW_SHA256) -> dict[str, Any]:
    return {
        "source_date": day,
        "archive_name": f"pokemon-tcg-ai-battle-episodes-{day}.zip",
        "sha256": sha256,
        "bytes": 123,
    }


def _runtime_code() -> routes.RuntimeCodeBinding:
    return routes.RuntimeCodeBinding(
        submission_bundle_sha256=BUNDLE_SHA256,
        submission_entrypoint_member="./main.py",
        submission_entrypoint_sha256=ENTRYPOINT_SHA256,
        public_matchup_router_member="./poke_bot/public_matchup_router.py",
        public_matchup_router_sha256=ROUTER_SHA256,
    )


def _producer_code() -> dict[str, str]:
    return routes.ProducerCodeBinding(
        guide2vec_public_routes_sha256=PRODUCER_ROUTE_SHA256,
        materializer_cli_sha256=PRODUCER_CLI_SHA256,
    ).as_dict()


def _write_feature_shard(
    directory: Path,
    sequence: GameSequence,
    *,
    day: str = DAY,
    raw_sha256: str = RAW_SHA256,
    name: str = "source.features",
) -> tuple[Path, str]:
    """Write the minimum valid temporal source/header consumed by the reader."""

    path = directory / name
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": "temporal-expert-v1",
        "required_archetype": "alakazam",
        "source_dates": [day],
        "source_archive": f"pokemon-tcg-ai-battle-episodes-{day}.zip",
        "source_archive_sha256": raw_sha256,
    }
    footer = {
        "format": SHARD_FORMAT + "-footer",
        "format_version": SHARD_FORMAT_VERSION,
        "stats": {"records_kept": 1},
    }
    with path.open("xb") as handle:
        pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sequence, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(footer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path, _sha256_file(path)


def _sidecar_parts(
    sequence: GameSequence,
    *,
    source_feature_shard_sha256: str,
    day: str = DAY,
    raw_archive: dict[str, Any] | None = None,
    game_resets: int = 0,
    game_reset_env_steps: tuple[int, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_archive = dict(raw_archive or _raw_archive(day=day))
    runtime_code = _runtime_code().as_dict()
    alignment = _alignment(sequence)
    result = routes.SequencePublicRoutes(
        episode_id=sequence.episode_id,
        seat=sequence.seat,
        env_steps=tuple(decision.env_step for decision in sequence.decisions),
        routes=(11,),
        compact_alignment_sha256=alignment,
        raw_member_sha256=RAW_MEMBER_SHA256,
        active_observations=1,
        turn_order_short_circuits=0,
        turn_order_short_circuit_env_steps=(),
        game_resets=game_resets,
        game_reset_env_steps=game_reset_env_steps,
    )
    row = result.as_sidecar_row()
    row_sha256 = "sha256:" + hashlib.sha256(routes._canonical_json(row)).hexdigest()
    header = {
        "schema": routes.ROUTE_SIDECAR_HEADER_SCHEMA,
        "format": routes.ROUTE_SIDECAR_FORMAT,
        "source_date": day,
        "source_feature_shard_sha256": source_feature_shard_sha256,
        "raw_archive": raw_archive,
        "runtime_public_tree_sha256": TREE_SHA256,
        "runtime_code": runtime_code,
        "producer_code": _producer_code(),
        "allowed_physical_slots": [11],
        "algorithm": routes.ROUTE_ALGORITHM,
        "alignment_contract": routes.ALIGNMENT_CONTRACT,
        "compact_source_routes_ignored": True,
        "oracle_route_used": False,
    }
    projection = {
        "schema": routes.ROUTE_RECONSTRUCTION_SCHEMA,
        "source_date": day,
        "source_feature_shard_sha256": source_feature_shard_sha256,
        "raw_archive": raw_archive,
        "runtime_public_tree_sha256": TREE_SHA256,
        "runtime_code": runtime_code,
        "producer_code": _producer_code(),
        "allowed_physical_slots": [11],
        "algorithm": routes.ROUTE_ALGORITHM,
        "alignment_contract": routes.ALIGNMENT_CONTRACT,
        "records": 1,
        "decisions": 1,
        "active_observations": 1,
        "turn_order_short_circuits": 0,
        "game_resets": game_resets,
        "routed_decisions": 1,
        "bypassed_decisions": 0,
        "member_route_sha256": row_sha256,
        "compact_source_routes_ignored": True,
        "oracle_route_used": False,
    }
    footer = {
        "schema": routes.ROUTE_SIDECAR_FOOTER_SCHEMA,
        "rows": 1,
        "rows_sha256": row_sha256,
        "projection": projection,
    }
    return header, row, footer


def _write_sidecar(
    directory: Path,
    *,
    header: dict[str, Any],
    row: dict[str, Any],
    footer: dict[str, Any],
    name: str = "sidecar.jsonl",
) -> tuple[Path, str]:
    path = directory / name
    path.write_text(
        "".join(
            routes._canonical_json(item).decode("utf-8")
            for item in (header, row, footer)
        ),
        encoding="utf-8",
    )
    return path, _sha256_file(path)


def _sidecar_kwargs(
    *,
    sidecar_path: Path,
    sidecar_sha256: str,
    feature_shard_path: Path,
    feature_shard_sha256: str,
    day: str = DAY,
    expected_raw_archive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sidecar_path": sidecar_path,
        "source_date": day,
        "feature_shard_path": feature_shard_path,
        "feature_shard_sha256": feature_shard_sha256,
        "expected_raw_archive": expected_raw_archive or _raw_archive(day=day),
        "matchup_tree_sha256": TREE_SHA256,
        "submission_bundle_sha256": BUNDLE_SHA256,
        "submission_entrypoint_member": "./main.py",
        "submission_entrypoint_sha256": ENTRYPOINT_SHA256,
        "public_matchup_router_member": "./poke_bot/public_matchup_router.py",
        "public_matchup_router_sha256": ROUTER_SHA256,
        "allowed_physical_slots": frozenset({11}),
        "expected_sidecar_sha256": sidecar_sha256,
        "expected_producer_code": _producer_code(),
    }


def _raw_episode(events: list[dict[str, Any]], *, episode_id: str = "episode-raw") -> bytes:
    """Build an official-shape two-seat trace with optional shifted actions."""

    steps: list[list[dict[str, Any]]] = []
    for event in events:
        observation = {
            "current": (
                None if event.get("current_null", False) else {"yourIndex": 0}
            )
        }
        observation["select"] = (
            None
            if event.get("select_null", False)
            else {
                "context": event.get("context", 1),
                "route": event.get("route", routes.UNKNOWN_ROUTE),
            }
        )
        seat_zero: dict[str, Any] = {
            "status": event.get("status", "INACTIVE"),
            "observation": observation,
        }
        if "action" in event:
            seat_zero["action"] = list(event["action"])
        steps.append([seat_zero, {"status": "INACTIVE"}])
    return json.dumps({"info": {"EpisodeId": episode_id}, "steps": steps}).encode("utf-8")


class _LastRouteRouter:
    """A scanner spy that makes each observed public route directly visible."""

    observed: ClassVar[list[int]] = []

    def __init__(self, _tree: object) -> None:
        self._route = routes.UNKNOWN_ROUTE

    @property
    def candidate_model_route(self) -> int:
        return self._route

    def observe(self, observation: dict[str, Any]) -> None:
        self._route = int(observation["select"]["route"])
        type(self).observed.append(self._route)

    def reset_for_new_game(self) -> None:
        self._route = routes.UNKNOWN_ROUTE


class _TwoObservationRouter:
    """Minimal exact-stability model of the router's two-observation rule."""

    def __init__(self, _tree: object) -> None:
        self._route = routes.UNKNOWN_ROUTE
        self._pending = routes.UNKNOWN_ROUTE
        self._pending_count = 0

    @property
    def candidate_model_route(self) -> int:
        return self._route

    def observe(self, observation: dict[str, Any]) -> None:
        route = int(observation["select"]["route"])
        if route == self._pending:
            self._pending_count += 1
        else:
            self._pending = route
            self._pending_count = 1
        if self._pending_count >= 2:
            self._route = route

    def reset_for_new_game(self) -> None:
        self._route = routes.UNKNOWN_ROUTE
        self._pending = routes.UNKNOWN_ROUTE
        self._pending_count = 0


def test_raw_scanner_ignores_inactive_stale_select_and_never_advances_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inactive historical selects are not submitted-agent calls."""

    _LastRouteRouter.observed = []
    monkeypatch.setattr(routes, "RuntimePublicMatchupRouter", _LastRouteRouter)
    seen_actions: list[list[int]] = []
    result = routes.reconstruct_public_routes_from_raw_member(
        raw_member=_raw_episode(
            [
                {"status": "INACTIVE", "route": 11},  # stale select
                {"status": "ACTIVE", "route": routes.UNKNOWN_ROUTE},
                {"status": "INACTIVE", "action": [7]},
            ]
        ),
        episode_id="episode-raw",
        seat=0,
        env_steps=[1],
        compact_alignment_sha256=RAW_MEMBER_SHA256,
        tree=object(),
        allowed_physical_slots={11},
        target_validator=lambda _index, _observation, action: seen_actions.append(action),
    )

    assert _LastRouteRouter.observed == [routes.UNKNOWN_ROUTE]
    assert seen_actions == [[7]]
    assert result.routes == (routes.UNKNOWN_ROUTE,)
    assert result.active_observations == 1


def test_raw_scanner_bypasses_isfirst_before_public_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r195 resolves IsFirst in main.py, without constructing a model route."""

    _LastRouteRouter.observed = []
    monkeypatch.setattr(routes, "RuntimePublicMatchupRouter", _LastRouteRouter)
    result = routes.reconstruct_public_routes_from_raw_member(
        raw_member=_raw_episode(
            [
                {"status": "ACTIVE", "context": "Is First", "route": 11},
                {"status": "INACTIVE", "action": [1]},
            ]
        ),
        episode_id="episode-raw",
        seat=0,
        env_steps=[0],
        compact_alignment_sha256=RAW_MEMBER_SHA256,
        tree=object(),
        allowed_physical_slots={11},
    )

    assert _LastRouteRouter.observed == []
    assert result.routes == (routes.UNKNOWN_ROUTE,)
    assert result.active_observations == 0
    assert result.turn_order_short_circuits == 1
    assert result.turn_order_short_circuit_env_steps == (0,)


def test_raw_scanner_preserves_two_observation_activation_and_deactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route only changes after the same public prediction is seen twice."""

    monkeypatch.setattr(routes, "RuntimePublicMatchupRouter", _TwoObservationRouter)
    result = routes.reconstruct_public_routes_from_raw_member(
        raw_member=_raw_episode(
            [
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": routes.UNKNOWN_ROUTE, "action": [4]},
                {"status": "ACTIVE", "route": routes.UNKNOWN_ROUTE},
                {"status": "INACTIVE", "action": [5]},
            ]
        ),
        episode_id="episode-raw",
        seat=0,
        env_steps=[1, 3],
        compact_alignment_sha256=RAW_MEMBER_SHA256,
        tree=object(),
        allowed_physical_slots={11},
    )

    assert result.routes == (11, routes.UNKNOWN_ROUTE)
    assert result.active_observations == 4
    assert result.turn_order_short_circuits == 0


def test_active_select_null_resets_router_but_inactive_stale_select_null_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The r195 entrypoint reset cannot leak a confirmed route into next game."""

    monkeypatch.setattr(routes, "RuntimePublicMatchupRouter", _TwoObservationRouter)
    reset_result = routes.reconstruct_public_routes_from_raw_member(
        raw_member=_raw_episode(
            [
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": 11},
                # Real r195 game-boundary rows carry neither a current state
                # nor legal selection data, but still invoke reset_game().
                {"status": "ACTIVE", "select_null": True, "current_null": True},
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": 11, "action": [5]},
                {"status": "INACTIVE", "action": [6]},
            ]
        ),
        episode_id="episode-raw",
        seat=0,
        # The retained compact window begins after reset.  A reset *inside*
        # a compact temporal sequence is correctly fail-closed instead.
        env_steps=[3, 4],
        compact_alignment_sha256=RAW_MEMBER_SHA256,
        tree=object(),
        allowed_physical_slots={11},
    )
    assert reset_result.routes == (routes.UNKNOWN_ROUTE, 11)
    assert reset_result.game_resets == 1
    assert reset_result.game_reset_env_steps == (2,)

    stale_result = routes.reconstruct_public_routes_from_raw_member(
        raw_member=_raw_episode(
            [
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": 11},
                {"status": "INACTIVE", "select_null": True, "current_null": True},
                {"status": "ACTIVE", "route": 11},
                {"status": "ACTIVE", "route": 11, "action": [5]},
                {"status": "INACTIVE", "action": [6]},
            ]
        ),
        episode_id="episode-raw",
        seat=0,
        env_steps=[3, 4],
        compact_alignment_sha256=RAW_MEMBER_SHA256,
        tree=object(),
        allowed_physical_slots={11},
    )
    assert stale_result.routes == (11, 11)
    assert stale_result.game_resets == 0
    assert stale_result.game_reset_env_steps == ()


def test_select_null_inside_a_retained_compact_window_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One temporal source sequence cannot silently span two r195 games."""

    monkeypatch.setattr(routes, "RuntimePublicMatchupRouter", _TwoObservationRouter)
    with pytest.raises(PublicRouteReconstructionError, match="span compact temporal history"):
        routes.reconstruct_public_routes_from_raw_member(
            raw_member=_raw_episode(
                [
                    {"status": "ACTIVE", "route": 11},
                    {"status": "ACTIVE", "route": 11},
                    {"status": "ACTIVE", "select_null": True, "action": [4]},
                    {"status": "ACTIVE", "route": 11},
                    {"status": "INACTIVE", "action": [5]},
                ]
            ),
            episode_id="episode-raw",
            seat=0,
            env_steps=[1, 3],
            compact_alignment_sha256=RAW_MEMBER_SHA256,
            tree=object(),
            allowed_physical_slots={11},
        )


def test_compact_alignment_rejects_shifted_board_and_action_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw target must reproduce both the board and shifted action token."""

    sequence = _sequence()
    sequence.decisions[0].board = _sparse(9)
    sequence.decisions[0].action_token = _sparse(7)

    monkeypatch.setattr(
        routes.features,
        "build_board_tokens",
        lambda observation, _deck: _sparse(int(observation["marker"])),
    )
    monkeypatch.setattr(
        routes.features,
        "build_option_tokens",
        lambda _observation, actions: _sparse(int(actions[0][0])),
    )
    observation = {
        "marker": 9,
        "current": {"yourIndex": 0, "players": [{}, {"hand": None}]},
    }

    routes._assert_compact_alignment(
        sequence, decision_index=0, observation=observation, raw_action=[7]
    )
    with pytest.raises(PublicRouteReconstructionError, match="board features"):
        routes._assert_compact_alignment(
            sequence,
            decision_index=0,
            observation={**observation, "marker": 8},
            raw_action=[7],
        )
    with pytest.raises(PublicRouteReconstructionError, match="action-token"):
        routes._assert_compact_alignment(
            sequence, decision_index=0, observation=observation, raw_action=[8]
        )


PublicRouteReconstructionError = routes.PublicRouteReconstructionError


def test_runtime_code_bundle_binding_requires_exact_entrypoint_and_router(
    tmp_path: Path,
) -> None:
    """The raw resolver cannot substitute a nearby entrypoint/router lookalike."""

    main = b"def agent(_observation, _configuration): return []\n"
    router = b"class RuntimePublicMatchupRouter: pass\n"
    bundle = tmp_path / "r195-submission.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, payload in (("./main.py", main), ("./poke_bot/public_matchup_router.py", router)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    bundle_sha256 = _sha256_file(bundle)
    main_sha256 = "sha256:" + hashlib.sha256(main).hexdigest()
    router_sha256 = "sha256:" + hashlib.sha256(router).hexdigest()

    binding, _stat = routes._bind_runtime_code(
        submission_bundle_path=bundle,
        submission_bundle_sha256=bundle_sha256,
        submission_entrypoint_member="./main.py",
        submission_entrypoint_sha256=main_sha256,
        public_matchup_router_member="./poke_bot/public_matchup_router.py",
        public_matchup_router_sha256=router_sha256,
    )
    assert binding.submission_entrypoint_sha256 == main_sha256
    assert binding.public_matchup_router_sha256 == router_sha256
    with pytest.raises(PublicRouteReconstructionError, match="entrypoint"):
        routes._bind_runtime_code(
            submission_bundle_path=bundle,
            submission_bundle_sha256=bundle_sha256,
            submission_entrypoint_member="./main.py",
            submission_entrypoint_sha256=ENTRYPOINT_SHA256,
            public_matchup_router_member="./poke_bot/public_matchup_router.py",
            public_matchup_router_sha256=router_sha256,
        )


def test_sidecar_reader_rejects_header_row_and_footer_tampering(tmp_path: Path) -> None:
    """The self-hashed JSONL cannot hide a runtime, row, or footer mutation."""

    sequence = _sequence(env_step=3)
    feature_shard, feature_sha256 = _write_feature_shard(tmp_path, sequence)
    header, row, footer = _sidecar_parts(
        sequence,
        source_feature_shard_sha256=feature_sha256,
        game_resets=1,
        game_reset_env_steps=(2,),
    )
    sidecar, sidecar_sha256 = _write_sidecar(
        tmp_path, header=header, row=row, footer=footer
    )
    kwargs = _sidecar_kwargs(
        sidecar_path=sidecar,
        sidecar_sha256=sidecar_sha256,
        feature_shard_path=feature_shard,
        feature_shard_sha256=feature_sha256,
    )
    with routes.SidecarPublicRouteResolver.open(**kwargs) as resolver:
        resolved = resolver.resolve_sequence(sequence)
        assert resolved.routes == (11,)
        assert resolved.game_resets == 1
        assert resolved.game_reset_env_steps == (2,)
        assert resolver.projection(expected_records=1)["game_resets"] == 1

    header_tampered = dict(header)
    header_tampered["runtime_code"] = dict(header_tampered["runtime_code"])
    header_tampered["runtime_code"]["submission_entrypoint_sha256"] = OTHER_RAW_SHA256
    header_sidecar, header_digest = _write_sidecar(
        tmp_path,
        header=header_tampered,
        row=row,
        footer=footer,
        name="header-tampered.jsonl",
    )
    with pytest.raises(PublicRouteReconstructionError, match="binding"):
        routes.SidecarPublicRouteResolver.open(
            **_sidecar_kwargs(
                sidecar_path=header_sidecar,
                sidecar_sha256=header_digest,
                feature_shard_path=feature_shard,
                feature_shard_sha256=feature_sha256,
            )
        )

    producer_tampered = dict(header)
    producer_tampered["producer_code"] = dict(producer_tampered["producer_code"])
    producer_tampered["producer_code"]["materializer_cli_sha256"] = OTHER_RAW_SHA256
    producer_sidecar, producer_digest = _write_sidecar(
        tmp_path,
        header=producer_tampered,
        row=row,
        footer=footer,
        name="producer-tampered.jsonl",
    )
    with pytest.raises(PublicRouteReconstructionError, match="binding"):
        routes.SidecarPublicRouteResolver.open(
            **_sidecar_kwargs(
                sidecar_path=producer_sidecar,
                sidecar_sha256=producer_digest,
                feature_shard_path=feature_shard,
                feature_shard_sha256=feature_sha256,
            )
        )

    row_tampered = dict(row)
    row_tampered["routes"] = [12]
    row_sidecar, row_digest = _write_sidecar(
        tmp_path,
        header=header,
        row=row_tampered,
        footer=footer,
        name="row-tampered.jsonl",
    )
    with (
        routes.SidecarPublicRouteResolver.open(
        **_sidecar_kwargs(
            sidecar_path=row_sidecar,
            sidecar_sha256=row_digest,
            feature_shard_path=feature_shard,
            feature_shard_sha256=feature_sha256,
        )
        )
        as resolver,
        pytest.raises(PublicRouteReconstructionError, match="invalid physical routes"),
    ):
        resolver.resolve_sequence(sequence)

    reset_row_tampered = dict(row)
    reset_row_tampered["game_reset_env_steps"] = []
    reset_sidecar, reset_digest = _write_sidecar(
        tmp_path,
        header=header,
        row=reset_row_tampered,
        footer=footer,
        name="reset-tampered.jsonl",
    )
    with (
        routes.SidecarPublicRouteResolver.open(
        **_sidecar_kwargs(
            sidecar_path=reset_sidecar,
            sidecar_sha256=reset_digest,
            feature_shard_path=feature_shard,
            feature_shard_sha256=feature_sha256,
        )
        )
        as resolver,
        pytest.raises(PublicRouteReconstructionError, match="accounting"),
    ):
        resolver.resolve_sequence(sequence)

    reset_window_tampered = dict(row)
    reset_window_tampered["game_reset_env_steps"] = [3]
    reset_window_sidecar, reset_window_digest = _write_sidecar(
        tmp_path,
        header=header,
        row=reset_window_tampered,
        footer=footer,
        name="reset-inside-window-tampered.jsonl",
    )
    with (
        routes.SidecarPublicRouteResolver.open(
        **_sidecar_kwargs(
            sidecar_path=reset_window_sidecar,
            sidecar_sha256=reset_window_digest,
            feature_shard_path=feature_shard,
            feature_shard_sha256=feature_sha256,
        )
        )
        as resolver,
        pytest.raises(PublicRouteReconstructionError, match="accounting"),
    ):
        resolver.resolve_sequence(sequence)

    footer_tampered = dict(footer)
    footer_tampered["rows_sha256"] = OTHER_RAW_SHA256
    footer_sidecar, footer_digest = _write_sidecar(
        tmp_path,
        header=header,
        row=row,
        footer=footer_tampered,
        name="footer-tampered.jsonl",
    )
    with routes.SidecarPublicRouteResolver.open(
        **_sidecar_kwargs(
            sidecar_path=footer_sidecar,
            sidecar_sha256=footer_digest,
            feature_shard_path=feature_shard,
            feature_shard_sha256=feature_sha256,
        )
    ) as resolver:
        resolver.resolve_sequence(sequence)
        with pytest.raises(PublicRouteReconstructionError, match="footer accounting"):
            resolver.projection(expected_records=1)


def test_sidecar_requires_digest_raw_archive_and_feature_header_binding(tmp_path: Path) -> None:
    """Consumer inputs must be SHA-bound to both the sidecar and protected shard."""

    sequence = _sequence()
    feature_shard, feature_sha256 = _write_feature_shard(tmp_path, sequence)
    header, row, footer = _sidecar_parts(
        sequence, source_feature_shard_sha256=feature_sha256
    )
    sidecar, sidecar_sha256 = _write_sidecar(
        tmp_path, header=header, row=row, footer=footer
    )
    kwargs = _sidecar_kwargs(
        sidecar_path=sidecar,
        sidecar_sha256=sidecar_sha256,
        feature_shard_path=feature_shard,
        feature_shard_sha256=feature_sha256,
    )
    missing_digest = dict(kwargs)
    missing_digest["expected_sidecar_sha256"] = None
    with pytest.raises(PublicRouteReconstructionError, match="sidecar SHA-256"):
        routes.SidecarPublicRouteResolver.open(**missing_digest)

    raw_mismatch = dict(kwargs)
    raw_mismatch["expected_raw_archive"] = _raw_archive(sha256=OTHER_RAW_SHA256)
    with pytest.raises(PublicRouteReconstructionError, match="raw archive binding"):
        routes.SidecarPublicRouteResolver.open(**raw_mismatch)

    changed_feature, changed_feature_sha256 = _write_feature_shard(
        tmp_path,
        sequence,
        raw_sha256=OTHER_RAW_SHA256,
        name="feature-header-tampered.features",
    )
    changed_header = dict(kwargs)
    changed_header["feature_shard_path"] = changed_feature
    changed_header["feature_shard_sha256"] = changed_feature_sha256
    changed_header["expected_raw_archive"] = _raw_archive(sha256=OTHER_RAW_SHA256)
    with pytest.raises(PublicRouteReconstructionError, match="binding"):
        routes.SidecarPublicRouteResolver.open(**changed_header)


def test_materializer_writes_one_checksum_bound_two_day_manifest(tmp_path: Path) -> None:
    """The Elmo transfer manifest keeps both heldout days and shared runtime code."""

    sidecars: dict[str, Path] = {}
    for day in (DAY, DAY_TWO):
        sequence = _sequence(episode_id=f"episode-{day}")
        shard, shard_sha256 = _write_feature_shard(
            tmp_path,
            sequence,
            day=day,
            raw_sha256=RAW_SHA256 if day == DAY else OTHER_RAW_SHA256,
            name=f"{day}.features",
        )
        del shard  # The manifest operates on transferred sidecars, not source bytes.
        header, row, footer = _sidecar_parts(
            sequence,
            source_feature_shard_sha256=shard_sha256,
            day=day,
            raw_archive=_raw_archive(
                day=day,
                sha256=RAW_SHA256 if day == DAY else OTHER_RAW_SHA256,
            ),
        )
        sidecar, _digest = _write_sidecar(
            tmp_path,
            header=header,
            row=row,
            footer=footer,
            name=f"{day}.jsonl",
        )
        sidecars[day] = sidecar

    result = materializer._write_manifest(
        argparse.Namespace(
            manifest_day=[f"{DAY}={sidecars[DAY]}", f"{DAY_TWO}={sidecars[DAY_TWO]}"],
            output_dir=tmp_path / "manifest",
        )
    )
    manifest_path = Path(result["manifest"])
    assert manifest_path.is_file()
    assert result["sha256"] == _sha256_file(manifest_path)
    assert result["payload"]["schema"] == materializer.SIDECAR_MANIFEST_SCHEMA
    assert result["payload"]["producer_code"] == _producer_code()
    assert tuple(result["payload"]["days"]) == (DAY, DAY_TWO)
    for day in (DAY, DAY_TWO):
        assert result["payload"]["days"][day]["path"] == sidecars[day].name
        assert result["payload"]["days"][day]["sha256"] == _sha256_file(sidecars[day])
