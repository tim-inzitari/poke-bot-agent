"""Belief particle priors: hand reweight, archetype soft prior, own prizes."""

from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

from poke_bot.belief import (
    EmpiricalDeckPosterior,
    NeuralBeliefPriors,
    PublicBeliefHistory,
)
from poke_bot.model import build_model, card_prior_logits_or_uniform
from poke_bot.train import belief_multihots_from_aux_labels, masked_belief_card_bce


def _card(card_id: int, serial: int, player: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def _observation(*, own_prize_known: dict[int, int] | None = None) -> dict:
    own_prizes = [None] * 6
    if own_prize_known:
        for slot, card_id in own_prize_known.items():
            own_prizes[slot] = _card(card_id, 200 + slot, 0)
    return {
        "search_begin_input": "opaque",
        "logs": [],
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
            "deck": None,
        },
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "result": -1,
            "stadium": [],
            "players": [
                {
                    "active": [_card(1, 6, 0)],
                    "bench": [],
                    "deckCount": 48,
                    "discard": [],
                    "prize": own_prizes,
                    "handCount": 5,
                    "hand": [_card(1, serial, 0) for serial in range(1, 6)],
                },
                {
                    "active": [_card(20, 100, 1)],
                    "bench": [],
                    "deckCount": 47,
                    "discard": [_card(10, 101, 1)],
                    "prize": [None] * 6,
                    "handCount": 5,
                    "hand": None,
                },
            ],
        },
    }


def test_own_prize_exact_tracking_and_illegal_reject() -> None:
    history = PublicBeliefHistory()
    obs = _observation(own_prize_known={0: 7, 2: 8})
    history.observe(obs)
    assert history.own_known_prizes == {0: 7, 2: 8}

    # Opp public evidence is cards 10 + 20; own known prizes need 7/8 in own deck.
    supporting = [1] * 18 + [10] * 20 + [20] * 20 + [7] * 1 + [8] * 1
    assert len(supporting) == 60
    posterior = EmpiricalDeckPosterior([supporting])
    good_own = [1] * 50 + [7] * 5 + [8] * 5
    particle = posterior.sample_particle(
        obs,
        own_deck=good_own,
        history=history,
        rng=random.Random(0),
    )
    assert particle.search_inputs["your_prize"][0] == 7
    assert particle.search_inputs["your_prize"][2] == 8

    # Own deck missing exact known prize 8 → conservation error.
    bad_own = [1] * 55 + [7] * 5
    with pytest.raises(Exception, match="violates visible|multiplicities"):
        posterior.sample_particle(
            obs,
            own_deck=bad_own,
            history=history,
            rng=random.Random(0),
        )


def test_archetype_soft_prior_moves_hypothesis_mass() -> None:
    # Both hypotheses must cover public opp cards 10 and 20.
    deck_a = [1] * 18 + [10] * 21 + [20] * 21  # archetype 0
    deck_b = [2] * 18 + [10] * 21 + [20] * 21  # archetype 1
    posterior = EmpiricalDeckPosterior(
        [deck_a, deck_b],
        deck_archetype_indices=[0, 1],
    )
    history = PublicBeliefHistory()
    history.observe(_observation())
    base = {h.digest: p for h, p in posterior.posterior(history)}
    neural = NeuralBeliefPriors(archetype_logits=(0.0, 5.0, 0.0))
    boosted = {h.digest: p for h, p in posterior.posterior(history, neural=neural)}
    # Find each hyp by archetype_index.
    by_arch = {h.archetype_index: h.digest for h, _ in posterior.posterior(history)}
    assert boosted[by_arch[1]] > base[by_arch[1]]
    assert boosted[by_arch[0]] < base[by_arch[0]]


def test_particle_conservation_holds_with_neural_priors() -> None:
    deck = [1] * 10 + [10] * 20 + [20] * 20 + [99] * 10
    posterior = EmpiricalDeckPosterior([deck])
    history = PublicBeliefHistory()
    obs = _observation()
    history.observe(obs)
    vocab = max(deck) + 3
    logits = [0.0] * vocab
    logits[99] = 3.0
    neural = NeuralBeliefPriors(
        opp_hand_logits=tuple(logits),
        opp_remainder_logits=tuple(logits),
    )
    particle = posterior.sample_particle(
        obs,
        own_deck=[1] * 60,
        history=history,
        rng=random.Random(1),
        neural=neural,
    )
    predicted = Counter()
    for key in ("opponent_deck", "opponent_prize", "opponent_hand", "opponent_active"):
        predicted.update(particle.search_inputs[key])
    # Public opp discard(10)+active(20) are outside prediction arrays; total
    # private fill + public evidence must reconstruct the hypothesis deck.
    public = Counter({10: 1, 20: 1})
    assert predicted + public == Counter(deck)


def test_weighted_hand_prefers_high_logit_legal_cards() -> None:
    # Preferred tech 99 dominates the private remainder after public 10/20.
    deck = [1] * 10 + [10] * 20 + [20] * 20 + [99] * 10
    assert len(deck) == 60
    posterior = EmpiricalDeckPosterior([deck])
    history = PublicBeliefHistory()
    obs = _observation()
    history.observe(obs)
    vocab = max(deck) + 3
    logits = [-4.0] * vocab
    logits[99] = 6.0
    neural = NeuralBeliefPriors(
        opp_hand_logits=tuple(logits),
        opp_remainder_logits=tuple(logits),
    )
    hits = 0
    trials = 40
    for seed in range(trials):
        particle = posterior.sample_particle(
            obs,
            own_deck=[1] * 60,
            history=history,
            rng=random.Random(seed),
            neural=neural,
        )
        if 99 in particle.search_inputs["opponent_hand"]:
            hits += 1
    assert hits >= 20


def test_uniform_fallback_helper_and_masked_aux_labels() -> None:
    uni = card_prior_logits_or_uniform(None, 10)
    assert torch.allclose(torch.softmax(uni, dim=-1), torch.full((10,), 0.1))
    hand, rem = belief_multihots_from_aux_labels(
        {
            "opp_hand": [{"id": 3}, {"id": 5}],
            "opp_deck_order": [7, 7],
            "opp_prizes": None,
        },
        card_vocab=16,
        device=torch.device("cpu"),
    )
    assert hand is not None and rem is not None
    assert float(hand[3]) == 1.0 and float(hand[5]) == 1.0
    assert float(rem[7]) == 1.0
    assert float(masked_belief_card_bce(torch.zeros(16), None)) == 0.0


def test_model_belief_heads_not_board_features() -> None:
    """Sanity: beliefs come from distinct heads on state_vec, not bag growth."""
    model = build_model(
        belief_card_vocab=32,
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    assert "opp_hand_head" in model.aux_heads_present
    assert model.opp_hand_head.out_features == 32
    # Board bag vocab unchanged by belief heads.
    assert model.encoder_vocab == 64
