from __future__ import annotations

from poke_bot import features
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence
from poke_bot.matchup_adapters import UNKNOWN_ROUTE, route_for_archetype
from poke_bot.matchup_adapter_activation import training_route_for_decision
from scripts.train_pure_rl import _ticket_dormant_matchup_adapter_sequences


def _digest(value: str) -> str:
    return "sha256:" + value * 64


def _sequence(
    episode: str,
    opponent: str,
    provenance: dict,
    *,
    acting_archetype: str = "alakazam",
) -> GameSequence:
    decision = DecisionSample(
        board=features.SparseVector(),
        options=features.SparseVector(),
        action=[0],
        action_combo_index=0,
        action_combos=[[0]],
        env_step=1,
    )
    return GameSequence(
        episode_id=episode,
        seat=0,
        archetype=acting_archetype,
        opp_archetype=opponent,
        deck=[],
        value=1.0,
        decisions=[decision],
        target_provenance=provenance,
    )


def test_live_ticketing_includes_lucario_and_mirror_only() -> None:
    lucario_package = _digest("a")
    gate = {
        "id": "alakazam-strong-public-lc55-test",
        "roster": [
            {
                "opponent_id": "lucario-public",
                "archetype_id": "lucario",
                "content_digest": lucario_package,
            }
        ],
    }
    lucario = _sequence(
        "public-lucario",
        "lucario",
        {
            "self_play": False,
            "collect": "strong_public_practice",
            "opponent_training_group": "strong_public_practice",
            "active_gate_id": gate["id"],
            "opponent_id": "lucario-public",
            "opponent_archetype_id": "lucario",
            "opponent_content_digest": lucario_package,
            "formal_eval": False,
        },
    )
    mirror = _sequence(
        "mirror",
        "alakazam",
        {
            "self_play": True,
            "collect": "self_play",
            "opponent_id": "self:iter20.pt",
            "opponent_archetype_id": "alakazam",
            "opponent_checkpoint_digest": _digest("b"),
        },
    )
    diverse = _sequence(
        "diverse",
        "lucario",
        {
            "self_play": False,
            "collect": "public_mix",
            "opponent_training_group": "diverse_public",
            "opponent_id": "unverified-lucario",
            "opponent_archetype_id": "lucario",
            "opponent_content_digest": _digest("c"),
        },
    )
    research = _sequence(
        "research",
        "lucario",
        {
            "self_play": False,
            "collect": "research_controls",
            "opponent_training_group": "research_controls",
            "opponent_id": "lucario-public",
            "opponent_archetype_id": "lucario",
            "opponent_content_digest": lucario_package,
        },
    )
    dataset = BootstrapDataset([lucario, mirror, diverse, research])

    receipt = _ticket_dormant_matchup_adapter_sequences(
        dataset,
        active_gate=gate,
        specialist_archetype="alakazam",
    )

    assert receipt["ticketed_sequences"] == 2
    assert receipt["route_sequences"] == {"lucario": 1, "alakazam": 1}
    assert route_for_archetype("lucario") == 8
    assert training_route_for_decision(lucario, lucario.decisions[0]) == 8
    assert training_route_for_decision(mirror, mirror.decisions[0]) == 7
    assert diverse.matchup_adapter_training_ticket == {}
    assert research.matchup_adapter_training_ticket == {}
    assert diverse.decisions[0].matchup_adapter_oracle_route == UNKNOWN_ROUTE
    assert all(
        row.decisions[0].matchup_adapter_public_route == UNKNOWN_ROUTE
        for row in (lucario, mirror, diverse, research)
    )


def test_live_ticketing_rejects_stale_gate_identity() -> None:
    package = _digest("d")
    gate = {
        "id": "current-gate",
        "roster": [
            {
                "opponent_id": "lucario-public",
                "archetype_id": "lucario",
                "content_digest": package,
            }
        ],
    }
    stale = _sequence(
        "stale",
        "lucario",
        {
            "self_play": False,
            "collect": "strong_public_practice",
            "opponent_training_group": "strong_public_practice",
            "active_gate_id": "old-gate",
            "opponent_id": "lucario-public",
            "opponent_archetype_id": "lucario",
            "opponent_content_digest": package,
        },
    )
    try:
        _ticket_dormant_matchup_adapter_sequences(
            BootstrapDataset([stale]),
            active_gate=gate,
            specialist_archetype="alakazam",
        )
    except RuntimeError as exc:
        assert "no exact" in str(exc)
    else:
        raise AssertionError("stale-gate public sequence was ticketed")


def test_live_ticketing_supports_trevenant_mirror() -> None:
    gate = {
        "id": "trevenant-specialist-gate",
        "roster": [
            {
                "opponent_id": "alakazam-public",
                "archetype_id": "alakazam",
                "content_digest": _digest("e"),
            }
        ],
    }
    mirror = _sequence(
        "trevenant-mirror",
        "hops-trevenant",
        {
            "self_play": True,
            "collect": "self_play",
            "opponent_id": "self:trevenant.pt",
            "opponent_archetype_id": "hops-trevenant",
            "opponent_checkpoint_digest": _digest("f"),
        },
        acting_archetype="hops-trevenant",
    )

    receipt = _ticket_dormant_matchup_adapter_sequences(
        BootstrapDataset([mirror]),
        active_gate=gate,
        specialist_archetype="hops-trevenant",
    )

    assert receipt["ticketed_sequences"] == 1
    assert receipt["route_sequences"] == {"hops-trevenant": 1}
    assert training_route_for_decision(
        mirror, mirror.decisions[0]
    ) == route_for_archetype("hops-trevenant")
