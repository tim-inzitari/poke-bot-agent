from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from replay_inspector.config import InspectorConfig
from replay_inspector.server import InspectorApplication, create_server
from replay_inspector.training_recipe import load_training_recipe_registry

CHECKPOINT_SHA256 = (
    "sha256:7480d81c54b1b98955108401fc04c82e93b6afe626a70c1b52fd467cc0cb704b"
)
STRATEGIC_WEIGHTS = {
    "action_q": 0.10,
    "action_type": 0.05,
    "action_target": 0.025,
    "action_resource": 0.025,
    "action_utility": 0.05,
    "tactical_outcome": 0.05,
    "opponent_response": 0.05,
    "resource_forecast": 0.025,
    "game_phase": 0.025,
    "outcome_distribution": 0.05,
    "remaining_turns": 0.025,
}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe() -> dict[str, object]:
    return {
        "scope": "source_backed_training_loss_multipliers",
        "training_only": True,
        "fine_tune_execution_authority": False,
        "evaluation_replays_training_eligible": False,
        "loss_weights": {
            "guide_strategic_directional_v2": 0.05,
            **STRATEGIC_WEIGHTS,
        },
        "strategic_head_loss_weights": STRATEGIC_WEIGHTS,
    }


def _write_registry(
    root: Path,
    *,
    checkpoint_sha256: str,
    duplicate: bool = False,
) -> Path:
    source = root / "sources" / "r175-contract.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"contract":"r175"}\n', encoding="utf-8")
    record = {
        "checkpoint_sha256": checkpoint_sha256,
        "status": "verified",
        "recipe": _recipe(),
        "evidence": [
            {
                "path": "sources/r175-contract.json",
                "sha256": _digest(source),
                "pointer": "/heads",
                "role": "r175_owner_training_contract",
            }
        ],
    }
    path = root / "ops" / "elmo" / "training-recipes.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.replay_model_inspector_training_recipe_registry/v1"
                ),
                "version": 1,
                "records": [record, record] if duplicate else [record],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_replay_archive(root: Path) -> tuple[int, int, Path]:
    submission_id = 77001
    episode_id = 88001
    directory = root / str(submission_id)
    directory.mkdir(parents=True)
    observation = {
        "current": {
            "turn": 2,
            "yourIndex": 0,
            "players": [{"hand": []}, {"hand": None}],
        },
        "select": {
            "context": "Hand",
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
                {
                    "status": "ACTIVE",
                    "observation": observation,
                    "action": [1],
                },
                {"status": "INACTIVE", "observation": {}, "action": []},
            ],
        ]
    }
    replay_path = directory / f"episode-{episode_id}-replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    (directory / "episodes.json").write_text(
        json.dumps(
            {
                "submission_id": submission_id,
                "episodes": [{"episode_id": episode_id, "own_agent": {"index": 0}}],
            }
        ),
        encoding="utf-8",
    )
    return submission_id, episode_id, replay_path


def _application(
    tmp_path: Path, *, recipe_checkpoint_sha256: str | None
) -> tuple[InspectorApplication, int, int]:
    archive = tmp_path / "archive"
    submission_id, episode_id, replay_path = _write_replay_archive(archive)
    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint = assets / "model.pt"
    bundle = assets / "bundle.tar.gz"
    checkpoint.write_bytes(b"checkpoint")
    bundle.write_bytes(b"bundle")
    manifest = tmp_path / "provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "poke_bot.replay_model_inspector_provenance/v1",
                "version": 1,
                "records": [
                    {
                        "submission_id": submission_id,
                        "status": "verified",
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": _digest(checkpoint),
                        },
                        "bundle": {"path": str(bundle), "sha256": _digest(bundle)},
                        "runtime": {
                            "matchup_tree_path": None,
                            "matchup_tree_sha256": None,
                        },
                        "replay": {
                            "games": [
                                {
                                    "episode_id": episode_id,
                                    "replay_sha256": _digest(replay_path),
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry_root = tmp_path / "recipe-registry-root"
    registry = _write_registry(
        registry_root,
        checkpoint_sha256=(
            _digest(checkpoint)
            if recipe_checkpoint_sha256 is None
            else recipe_checkpoint_sha256
        ),
    )
    config = InspectorConfig(
        replay_root=archive,
        rollout_root=tmp_path / "rollouts",
        provenance_manifest=manifest,
        training_recipe_registry=registry,
        artifact_roots=(assets,),
        web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
    )
    return InspectorApplication(config), submission_id, episode_id


def _request(base_url: str, path: str) -> tuple[int, dict[str, object]]:
    try:
        with urlopen(Request(f"{base_url}{path}"), timeout=5) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return int(error.code), json.loads(error.read().decode("utf-8"))


def test_checked_in_r187_recipe_is_exactly_keyed_to_alakazam_checkpoint() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "elmo"
        / "replay-model-inspector-training-recipes-r187.json"
    )
    registry = load_training_recipe_registry(path)

    record, reasons = registry.resolve_checkpoint(CHECKPOINT_SHA256)

    assert reasons == ()
    assert record is not None
    assert record.available is True
    assert record.recipe is not None
    assert record.recipe["loss_weights"] == {
        "policy_cross_entropy": 1.0,
        "value": 1.0,
        "archetype": 0.05,
        "opponent_hand": 0.05,
        "opponent_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "guide_strategic_directional_v2": 0.05,
        "setup_board_outcome": 0.025,
        "combo_state": 0.0,
        **STRATEGIC_WEIGHTS,
    }
    assert record.recipe["strategic_head_loss_weights"] == STRATEGIC_WEIGHTS
    assert all(item.available for item in record.evidence)


def test_registry_rejects_ambiguous_checkpoint_recipe_mapping(tmp_path: Path) -> None:
    registry = load_training_recipe_registry(
        _write_registry(
            tmp_path / "repo", checkpoint_sha256=CHECKPOINT_SHA256, duplicate=True
        )
    )

    record, reasons = registry.resolve_checkpoint(CHECKPOINT_SHA256)

    assert record is None
    assert reasons == ("training_recipe_checkpoint_mapping_ambiguous",)


def test_recipe_endpoint_and_catalog_fields_fail_closed_when_unmapped(
    tmp_path: Path,
) -> None:
    # The provenance checkpoint has a deliberately different canonical digest
    # from the recipe registry.  The server must not substitute the only
    # available recipe based on submission identity or a filename.
    different_digest = "sha256:" + "a" * 64
    application, submission_id, episode_id = _application(
        tmp_path, recipe_checkpoint_sha256=different_digest
    )
    server = create_server(application.config, application=application, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        status, submissions = _request(base_url, "/api/submissions")
        assert status == 200
        recipe = submissions["submissions"][0]["training_recipe"]
        assert recipe["status"] == "unavailable"
        assert recipe["availability"] == {
            "available": False,
            "reason": "training_recipe_checkpoint_mapping_unavailable",
        }
        assert recipe["recipe"] is None

        status, payload = _request(
            base_url, f"/api/submissions/{submission_id}/training-recipe"
        )
        assert status == 200
        assert payload["training_recipe"] == recipe

        status, games = _request(base_url, f"/api/submissions/{submission_id}/games")
        assert status == 200
        assert games["training_recipe"] == recipe

        status, trace = _request(
            base_url,
            f"/api/submissions/{submission_id}/games/{episode_id}/steps/0?stage=0",
        )
        assert status == 200
        assert trace["training_recipe"] == recipe
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_recipe_is_passed_through_without_loading_or_modifying_a_model(
    tmp_path: Path,
) -> None:
    application, submission_id, _episode_id = _application(
        tmp_path, recipe_checkpoint_sha256=None
    )

    catalog_recipe = application.submissions_payload()["submissions"][0][
        "training_recipe"
    ]
    endpoint_recipe = application.training_recipe_payload(submission_id)[
        "training_recipe"
    ]

    assert endpoint_recipe == catalog_recipe
    assert endpoint_recipe["status"] == "available"
    assert endpoint_recipe["availability"] == {"available": True}
    assert (
        endpoint_recipe["recipe"]["loss_weights"]["guide_strategic_directional_v2"]
        == 0.05
    )
