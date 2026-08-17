from __future__ import annotations

from poke_bot.public_matchup_router import (
    PublicMatchupDecisionTree,
    RuntimePublicMatchupRouter,
    public_matchup_from_observation,
    visible_opponent_card_ids,
)
from poke_bot.matchup_adapters import EXPERT_IDS, UNKNOWN_ROUTE
from poke_bot.matchup_adapters_v6 import load_slot_registry
import json
import pytest


def _obs(*, your_index: int = 0) -> dict:
    return {
        "current": {
            "yourIndex": your_index,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "hand": [],
                    "deck": [265],
                    "prize": [265],
                },
                {
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "hand": None,
                    "deck": [265],
                    "prize": [265],
                },
            ],
        },
        "opponent_id": "iono",
        "opp_archetype": "iono",
    }


def test_router_abstains_when_iono_exists_only_in_hidden_or_metadata_fields() -> None:
    obs = _obs()
    obs["current"]["players"][1]["hand"] = [{"id": 265}]

    assert visible_opponent_card_ids(obs) == frozenset()
    assert public_matchup_from_observation(obs) is None


def test_router_uses_only_the_opponent_public_zones() -> None:
    obs = _obs()
    obs["current"]["players"][0]["active"] = [{"id": 269}]
    assert public_matchup_from_observation(obs) is None

    obs["current"]["players"][1]["active"] = [{"id": 265}]
    assert public_matchup_from_observation(obs) == "iono"


def test_router_recognizes_public_evolution_and_discard_evidence() -> None:
    obs = _obs(your_index=1)
    obs["current"]["players"][0]["bench"] = [
        {"id": 999, "preEvolution": [{"id": 268}]}
    ]
    assert public_matchup_from_observation(obs) == "iono"

    obs = _obs()
    obs["current"]["players"][1]["discard"] = [{"id": 271}]
    assert public_matchup_from_observation(obs) == "iono"


def test_router_does_not_treat_ns_joltik_as_iono() -> None:
    obs = _obs()
    obs["current"]["players"][1]["active"] = [{"id": 267}]
    assert public_matchup_from_observation(obs) is None


def test_router_fails_closed_on_malformed_observations() -> None:
    for obs in (None, {}, {"current": {}}, {"current": {"yourIndex": 3}}):
        assert visible_opponent_card_ids(obs) == frozenset()
        assert public_matchup_from_observation(obs) is None


def test_exported_tree_has_exact_canonical_route_positions_and_separate_abstention(
    tmp_path,
) -> None:
    width = len(EXPERT_IDS) + 1
    unknown = [0.0] * width
    unknown[-1] = 1.0
    crustle = [0.0] * width
    crustle[0] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": list(EXPERT_IDS),
        "prediction_contract": {
            "route_output_width": len(EXPERT_IDS),
            "route_class_names": list(EXPERT_IDS),
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(EXPERT_IDS),
            "adapter_count": len(EXPERT_IDS),
        },
        "tree": {
            "class_names": [*EXPERT_IDS, "unknown"],
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature_card_id": [42, -2, -2],
            "threshold": [0.5, -2.0, -2.0],
            "weighted_class_counts": [unknown, unknown, crustle],
            "node_count": 3,
            "max_depth": 1,
        },
    }
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(payload))
    tree = PublicMatchupDecisionTree.from_path(
        path, require_runtime_enabled=False
    )

    assert len(tree.targets) == len(EXPERT_IDS) == 18
    assert tree.predict_card_ids([]).route == UNKNOWN_ROUTE
    prediction = tree.predict_card_ids([42])
    assert prediction.route == 0
    assert prediction.archetype_id == "crustle"
    assert prediction.confidence == 1.0


def test_exported_tree_accepts_the_checksum_validated_v6_route_order() -> None:
    targets = list(load_slot_registry()["active_expert_ids"])
    width = len(targets) + 1
    teal = [0.0] * width
    teal[targets.index("teal-mask-ogerpon-ex")] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": targets,
        "prediction_contract": {
            "route_output_width": len(targets),
            "route_class_names": targets,
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(targets),
            "adapter_count": len(targets),
        },
        "tree": {
            "class_names": [*targets, "unknown"],
            "children_left": [-1],
            "children_right": [-1],
            "feature_card_id": [-2],
            "threshold": [-2.0],
            "weighted_class_counts": [teal],
            "node_count": 1,
        },
    }

    tree = PublicMatchupDecisionTree(
        payload,
        digest="sha256:" + "0" * 64,
    )

    assert len(tree.targets) == len(targets)
    assert tree.targets[:18] == EXPERT_IDS
    prediction = tree.predict_card_ids([96])
    assert prediction.archetype_id == "teal-mask-ogerpon-ex"
    assert prediction.route == 18


def test_exported_tree_accepts_only_an_append_only_v6_registry_prefix() -> None:
    registry = load_slot_registry()
    targets = list(registry["active_expert_ids"][:-1])
    width = len(targets) + 1
    unknown = [0.0] * width
    unknown[-1] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": targets,
        "prediction_contract": {
            "route_output_width": len(targets),
            "route_class_names": targets,
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(targets),
            "adapter_count": len(targets),
        },
        "runtime_contract": {
            "slot_registry_digest": "sha256:" + "1" * 64,
        },
        "tree": {
            "class_names": [*targets, "unknown"],
            "children_left": [-1],
            "children_right": [-1],
            "feature_card_id": [-2],
            "threshold": [-2.0],
            "weighted_class_counts": [unknown],
            "node_count": 1,
        },
    }
    tree = PublicMatchupDecisionTree(
        payload,
        digest="sha256:" + "0" * 64,
    )
    assert tree.targets == tuple(targets)

    payload["targets"] = [*targets[:-1], registry["active_expert_ids"][-1]]
    payload["prediction_contract"]["route_class_names"] = payload["targets"]
    payload["tree"]["class_names"] = [*payload["targets"], "unknown"]
    with pytest.raises(ValueError, match="canonical V5 or V6"):
        PublicMatchupDecisionTree(
            payload,
            digest="sha256:" + "0" * 64,
        )


def test_runtime_router_starts_dormant_and_continuously_corrects_route() -> None:
    width = len(EXPERT_IDS) + 1
    crustle = [0.0] * width
    crustle[0] = 1.0
    lucario_route = EXPERT_IDS.index("lucario")
    lucario = [0.0] * width
    lucario[lucario_route] = 1.0
    unknown = [0.0] * width
    unknown[-1] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": True,
        "targets": list(EXPERT_IDS),
        "prediction_contract": {
            "route_output_width": len(EXPERT_IDS),
            "route_class_names": list(EXPERT_IDS),
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(EXPERT_IDS),
            "adapter_count": len(EXPERT_IDS),
        },
        "runtime_contract": {
            "accepted_archetype_ids": ["crustle", "lucario"],
            "per_archetype_min_leaf_confidence": {
                "crustle": 0.9,
                "lucario": 0.9,
            },
            "min_leaf_confidence": 0.9,
            "consecutive_required": 2,
            "unknown_route_exact_bypass": True,
            "one_route_per_decision": True,
        },
        "tree": {
            "class_names": [*EXPERT_IDS, "unknown"],
            "children_left": [1, -1, 3, -1, -1],
            "children_right": [2, -1, 4, -1, -1],
            "feature_card_id": [42, -2, 84, -2, -2],
            "threshold": [0.5, -2.0, 0.5, -2.0, -2.0],
            "weighted_class_counts": [unknown, unknown, unknown, crustle, lucario],
            "node_count": 5,
        },
    }
    router = RuntimePublicMatchupRouter(
        PublicMatchupDecisionTree(payload, digest="sha256:" + "0" * 64)
    )

    assert router.candidate_model_route == UNKNOWN_ROUTE
    crustle_obs = _obs()
    crustle_obs["current"]["players"][1]["active"] = [{"id": 42}]
    router.observe(crustle_obs)
    assert router.candidate_model_route == UNKNOWN_ROUTE
    router.observe(crustle_obs)
    assert router.candidate_model_route == 0

    # One contradictory reading cannot flap the enabled route.
    router._public_card_ids.clear()
    lucario_obs = _obs()
    lucario_obs["current"]["players"][1]["active"] = [
        {"id": 42},
        {"id": 84},
    ]
    router.observe(lucario_obs)
    assert router.candidate_model_route == 0
    router.observe(lucario_obs)
    assert router.candidate_model_route == lucario_route

    # Repeated abstention returns all adapters to dormant exact-bypass.
    router._public_card_ids.clear()
    router.observe(_obs())
    assert router.candidate_model_route == lucario_route
    router.observe(_obs())
    assert router.candidate_model_route == UNKNOWN_ROUTE
    snapshot = router.snapshot()
    assert snapshot["active_archetype_id"] is None
    assert snapshot["accepted_routes"] == {
        "crustle": 0,
        "lucario": lucario_route,
    }
    assert snapshot["runtime_enabled"] is True
    assert snapshot["observations"] == 6
    assert snapshot["initial_model_route"] == UNKNOWN_ROUTE
    assert snapshot["route_transition_count"] == 3
    assert snapshot["route_transitions"] == [
        {"observation": 2, "from_route": UNKNOWN_ROUTE, "to_route": 0},
        {"observation": 4, "from_route": 0, "to_route": lucario_route},
        {
            "observation": 6,
            "from_route": lucario_route,
            "to_route": UNKNOWN_ROUTE,
        },
    ]
    assert router.audit is router
    assert router.audit.snapshot(include_events=False)["route_transitions"] == []


def test_runtime_tree_exactly_bypasses_empty_public_state_even_if_leaf_routes() -> None:
    width = len(EXPERT_IDS) + 1
    routed = [0.0] * width
    routed[0] = 1.0
    payload = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": True,
        "targets": list(EXPERT_IDS),
        "prediction_contract": {
            "route_output_width": len(EXPERT_IDS),
            "route_class_names": list(EXPERT_IDS),
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(EXPERT_IDS),
            "adapter_count": len(EXPERT_IDS),
        },
        "runtime_contract": {
            "accepted_archetype_ids": ["crustle"],
            "per_archetype_min_leaf_confidence": {"crustle": 0.9},
            "min_leaf_confidence": 0.9,
            "consecutive_required": 1,
            "unknown_route_exact_bypass": True,
            "one_route_per_decision": True,
        },
        "tree": {
            "class_names": [*EXPERT_IDS, "unknown"],
            "children_left": [-1],
            "children_right": [-1],
            "feature_card_id": [-2],
            "threshold": [-2.0],
            "weighted_class_counts": [routed],
            "node_count": 1,
        },
    }
    tree = PublicMatchupDecisionTree(payload, digest="sha256:" + "0" * 64)

    assert tree.predict_card_ids([]).route == 0
    prediction = tree.runtime_prediction([])
    assert prediction.archetype_id is None
    assert prediction.route == UNKNOWN_ROUTE
    assert prediction.confidence == 0.0
    assert prediction.leaf == -1
