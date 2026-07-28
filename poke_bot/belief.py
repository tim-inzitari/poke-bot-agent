"""Deployment-safe empirical deck posterior and hidden-state particles.

The posterior is fit from an anonymous corpus of deck lists. Runtime updates use
only the acting agent's observation history: public opponent cards and the
agent's own private state. Baseline IDs and the true opponent hidden zones are
never accepted by this interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import paths


BELIEF_SCHEMA_VERSION = 1
_BASIC_POKEMON_IDS: Optional[frozenset[int]] = None


class BeliefSupportError(RuntimeError):
    """No anonymous deck hypothesis conserves the observed public cards."""


def _basic_pokemon_ids() -> frozenset[int]:
    global _BASIC_POKEMON_IDS
    if _BASIC_POKEMON_IDS is None:
        from . import cg_env

        _BASIC_POKEMON_IDS = frozenset(
            int(card.cardId)
            for card in cg_env.all_card_data()
            if bool(getattr(card, "basic", False))
        )
    return _BASIC_POKEMON_IDS


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    peak = max(float(x) for x in logits)
    exps = [math.exp(float(x) - peak) for x in logits]
    normalizer = sum(exps) or 1.0
    return [value / normalizer for value in exps]


def _weighted_sample_without_replacement(
    pool: list[int],
    k: int,
    *,
    logit_lookup: Optional[Sequence[float]],
    rng: random.Random,
) -> list[int]:
    """Draw ``k`` cards from ``pool``; logits reweight legal multiset only."""
    if k < 0 or k > len(pool):
        raise BeliefSupportError("weighted sample size out of range for pool")
    if k == 0:
        return []
    remaining = list(pool)
    if logit_lookup is None:
        rng.shuffle(remaining)
        return remaining[:k]
    chosen: list[int] = []
    for _ in range(k):
        weights: list[float] = []
        for card_id in remaining:
            if 0 <= int(card_id) < len(logit_lookup):
                weights.append(math.exp(float(logit_lookup[int(card_id)])))
            else:
                weights.append(1.0)
        pick = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        chosen.append(remaining.pop(pick))
    return chosen


def _card_id(card: Any) -> Optional[int]:
    if not isinstance(card, dict):
        return None
    value = card.get("id")
    return int(value) if value is not None else None


def _walk_card(
    card: Any,
    *,
    seen_serials: set[tuple[int, int]],
    out: Counter[int],
) -> None:
    if not isinstance(card, dict):
        return
    card_id = _card_id(card)
    serial = int(card.get("serial", -1))
    player = int(card.get("playerIndex", -1))
    identity = (player, serial)
    if card_id is not None and (serial < 0 or identity not in seen_serials):
        out[card_id] += 1
        if serial >= 0:
            seen_serials.add(identity)
    for key in ("energyCards", "tools", "preEvolution"):
        for nested in card.get(key) or []:
            _walk_card(nested, seen_serials=seen_serials, out=out)


def _state(obs: dict[str, Any]) -> tuple[dict[str, Any], int]:
    current = obs.get("current")
    if not isinstance(current, dict):
        raise ValueError("belief sampling requires a post-setup observation")
    seat = int(current.get("yourIndex", -1))
    if seat not in (0, 1):
        raise ValueError("invalid acting seat in observation")
    players = current.get("players") or []
    if len(players) != 2:
        raise ValueError("observation must contain two player states")
    return current, seat


def assert_deployment_observation(obs: dict[str, Any]) -> None:
    """Reject any decoded opponent-private field before belief/search use."""
    current, seat = _state(obs)
    opponent = current["players"][1 - seat]
    if opponent.get("hand") is not None:
        raise ValueError("hidden-state leakage: opponent hand is visible")
    forbidden = (
        "deck",
        "deckOrder",
        "deck_order",
        "prizeOrder",
        "prize_cards",
        "hiddenPrize",
        "truePrize",
        "privateState",
    )
    leaked = [key for key in forbidden if opponent.get(key) is not None]
    if leaked:
        raise ValueError(
            f"hidden-state leakage: opponent privileged fields {leaked}"
        )


def _outside_prediction_counter(
    obs: dict[str, Any],
    player_index: int,
) -> Counter[int]:
    """Cards serialized outside deck/prize/opponent-hand prediction arrays."""
    current, _ = _state(obs)
    player = current["players"][player_index]
    out: Counter[int] = Counter()
    seen: set[tuple[int, int]] = set()
    for card in player.get("discard") or []:
        _walk_card(card, seen_serials=seen, out=out)
    for card in player.get("active") or []:
        _walk_card(card, seen_serials=seen, out=out)
    for card in player.get("bench") or []:
        _walk_card(card, seen_serials=seen, out=out)
    # The acting player's own hand is serialized and must not be predicted.
    for card in player.get("hand") or []:
        _walk_card(card, seen_serials=seen, out=out)
    for card in current.get("stadium") or []:
        if isinstance(card, dict) and int(card.get("playerIndex", -1)) == player_index:
            _walk_card(card, seen_serials=seen, out=out)
    for card in current.get("looking") or []:
        if isinstance(card, dict) and int(card.get("playerIndex", -1)) == player_index:
            _walk_card(card, seen_serials=seen, out=out)
    for card in (obs.get("select") or {}).get("deck") or []:
        if isinstance(card, dict) and int(card.get("playerIndex", -1)) == player_index:
            _walk_card(card, seen_serials=seen, out=out)
    select = obs.get("select") or {}
    for key in ("contextCard", "effect"):
        card = select.get(key)
        if isinstance(card, dict) and int(card.get("playerIndex", -1)) == player_index:
            _walk_card(card, seen_serials=seen, out=out)
    return out


def _visible_prize_counter(player: dict[str, Any]) -> Counter[int]:
    out: Counter[int] = Counter()
    seen: set[tuple[int, int]] = set()
    for card in player.get("prize") or []:
        _walk_card(card, seen_serials=seen, out=out)
    return out


def _public_opponent_evidence(obs: dict[str, Any]) -> list[tuple[int, int]]:
    """Return unique ``(serial, card_id)`` public opponent cards in this view."""
    current, seat = _state(obs)
    opponent = 1 - seat
    player = current["players"][opponent]
    cards: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def visit(card: Any) -> None:
        if not isinstance(card, dict):
            return
        card_id = _card_id(card)
        serial = int(card.get("serial", -1))
        owner = int(card.get("playerIndex", opponent))
        identity = (owner, serial)
        if (
            card_id is not None
            and owner == opponent
            and (serial < 0 or identity not in seen)
        ):
            cards.append((serial, card_id))
            if serial >= 0:
                seen.add(identity)
        for key in ("energyCards", "tools", "preEvolution"):
            for nested in card.get(key) or []:
                visit(nested)

    for key in ("discard", "active", "bench", "prize"):
        for card in player.get(key) or []:
            visit(card)
    for card in current.get("stadium") or []:
        visit(card)
    return cards


@dataclass(frozen=True)
class NeuralBeliefPriors:
    """Root-only NN priors (not board features).

    Scope A: particle importance weights (archetype / hand / remainder).
    Scope B (Blackwell Hammer, gated): lethal_threat logit + prize_race
    prediction for optional root-only value bias — never board-bag injection.

    Missing / untrained Scope A card heads → caller should pass ``None`` so
    sampling falls back to uniform over the legal multiset.
    """

    archetype_logits: Optional[tuple[float, ...]] = None
    opp_hand_logits: Optional[tuple[float, ...]] = None
    opp_remainder_logits: Optional[tuple[float, ...]] = None
    #: Scope B — scalar logit for P(take prize soon); unused on core.
    lethal_threat_logit: Optional[float] = None
    #: Scope B — ``(own/6, opp/6)`` prize-race scaffold; unused on core.
    prize_race: Optional[tuple[float, float]] = None
    uniform_fallback: bool = False


@dataclass
class PublicBeliefHistory:
    """Monotone public-evidence tracker keyed by engine card serial."""

    observed_by_serial: dict[int, int] = field(default_factory=dict)
    unkeyed_max_counts: Counter[int] = field(default_factory=Counter)
    observation_digests: list[str] = field(default_factory=list)
    self_facedown_active_card: Optional[int] = None
    #: Exact own prize-slot → cardId from prize-checks / visible own prizes.
    #: Never inferred by NN; only exact self information.
    own_known_prizes: dict[int, int] = field(default_factory=dict)

    def observe(self, obs: dict[str, Any]) -> None:
        # Hard hidden-field guard. The opponent hand must stay absent.
        assert_deployment_observation(obs)
        current, seat = _state(obs)
        own_active = list(current["players"][seat].get("active") or [])
        if own_active and own_active[0] is None:
            for log in reversed(obs.get("logs") or []):
                if (
                    isinstance(log, dict)
                    and int(log.get("playerIndex", -1)) == seat
                    and int(log.get("toArea", -1)) == 4
                    and log.get("cardId") is not None
                ):
                    self.self_facedown_active_card = int(log["cardId"])
                    break
        elif own_active and _card_id(own_active[0]) is not None:
            self.self_facedown_active_card = None
        self._update_own_known_prizes(obs)
        for serial, card_id in _public_opponent_evidence(obs):
            if serial >= 0:
                old = self.observed_by_serial.get(serial)
                if old is not None and old != card_id:
                    raise ValueError("public card serial changed identity")
                self.observed_by_serial[serial] = card_id
            else:
                current_count = sum(
                    1
                    for seen_serial, seen_id in _public_opponent_evidence(obs)
                    if seen_serial < 0 and seen_id == card_id
                )
                self.unkeyed_max_counts[card_id] = max(
                    self.unkeyed_max_counts[card_id], current_count
                )
        public_projection = {
            "turn": current.get("turn"),
            "yourIndex": seat,
            "evidence": sorted(self.evidence_counter().items()),
            "own_known_prizes": sorted(self.own_known_prizes.items()),
            "select": {
                "type": (obs.get("select") or {}).get("type"),
                "context": (obs.get("select") or {}).get("context"),
                "minCount": (obs.get("select") or {}).get("minCount"),
                "maxCount": (obs.get("select") or {}).get("maxCount"),
                "optionCount": len((obs.get("select") or {}).get("option") or []),
            },
        }
        self.observation_digests.append(_digest(public_projection))

    def _update_own_known_prizes(self, obs: dict[str, Any]) -> None:
        """Exact prize-check / revealed own prize tracking (belief code, not NN)."""
        current, seat = _state(obs)
        prizes = list(current["players"][seat].get("prize") or [])
        for slot, card in enumerate(prizes):
            card_id = _card_id(card)
            if card_id is None:
                continue
            old = self.own_known_prizes.get(slot)
            if old is not None and old != card_id:
                raise ValueError(
                    f"own prize slot {slot} changed identity {old} → {card_id}"
                )
            self.own_known_prizes[slot] = card_id
        # Deck-search / looking conservation: cards the acting seat just viewed
        # that belong to own prize zone via log toArea prize (area code 5).
        for log in obs.get("logs") or []:
            if not isinstance(log, dict):
                continue
            if int(log.get("playerIndex", -1)) != seat:
                continue
            if int(log.get("toArea", -1)) != 5 and int(log.get("fromArea", -1)) != 5:
                continue
            if log.get("cardId") is None:
                continue
            slot = log.get("prizeIndex", log.get("index"))
            if slot is None:
                continue
            slot_i = int(slot)
            card_id = int(log["cardId"])
            old = self.own_known_prizes.get(slot_i)
            if old is not None and old != card_id:
                raise ValueError(
                    f"own prize slot {slot_i} changed identity {old} → {card_id}"
                )
            self.own_known_prizes[slot_i] = card_id

    def record_action(self, obs: dict[str, Any], selected: Sequence[int]) -> None:
        """Remember self-private setup choices needed by later masked views."""
        current, seat = _state(obs)
        player = current["players"][seat]
        active = list(player.get("active") or [])
        if active:
            return
        select = obs.get("select") or {}
        options = list(select.get("option") or [])
        hand = list(player.get("hand") or [])
        if not selected:
            return
        first_index = int(selected[0])
        if not 0 <= first_index < len(options):
            return
        option = options[first_index] or {}
        # Setup options identify a Basic by its hand index. The acting player
        # selected this card before the API masks its face-down active slot.
        if int(option.get("area", -1)) != 2:
            return
        hand_index = int(option.get("index", -1))
        if 0 <= hand_index < len(hand):
            card_id = _card_id(hand[hand_index])
            if card_id is not None:
                self.self_facedown_active_card = card_id

    def evidence_counter(self) -> Counter[int]:
        out = Counter(self.observed_by_serial.values())
        for card_id, count in self.unkeyed_max_counts.items():
            out[card_id] = max(out[card_id], count)
        return out

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema": BELIEF_SCHEMA_VERSION,
                "cards": sorted(self.evidence_counter().items()),
                "observations": self.observation_digests,
                "self_facedown_active_card": self.self_facedown_active_card,
                "own_known_prizes": sorted(self.own_known_prizes.items()),
            }
        )


@dataclass(frozen=True)
class DeckHypothesis:
    cards: tuple[int, ...]
    digest: str
    prior_count: int
    #: Optional taxonomy index for ``aux_head`` soft reweight (not baseline ID).
    archetype_index: Optional[int] = None


@dataclass(frozen=True)
class HiddenStateParticle:
    search_inputs: dict[str, list[int]]
    opponent_deck: tuple[int, ...]
    opponent_deck_digest: str
    belief_model_digest: str
    public_history_digest: str
    posterior_probability: float
    support_mode: str = "empirical_exact"
    support_repairs: int = 0


class EmpiricalDeckPosterior:
    """Anonymous empirical-Bayes deck posterior learned from deck-list counts."""

    sampler_name = "empirical-public-deck-posterior-v2"

    def __init__(
        self,
        deck_lists: Sequence[Sequence[int]],
        *,
        deck_archetype_indices: Optional[Sequence[Optional[int]]] = None,
    ) -> None:
        if not deck_lists:
            raise ValueError("belief posterior requires at least one deck list")
        if deck_archetype_indices is not None and len(deck_archetype_indices) != len(
            deck_lists
        ):
            raise ValueError("deck_archetype_indices length must match deck_lists")
        counts: Counter[tuple[int, ...]] = Counter()
        arch_votes: dict[tuple[int, ...], Optional[int]] = {}
        for i, raw in enumerate(deck_lists):
            deck = tuple(int(card) for card in raw)
            if len(deck) != 60:
                raise ValueError("every posterior deck hypothesis must have 60 cards")
            # Card order is not an identity signal for a shuffled deck.
            key = tuple(sorted(deck))
            counts[key] += 1
            if deck_archetype_indices is not None:
                # Keep first non-None taxonomy label if duplicates conflict.
                label = deck_archetype_indices[i]
                prev = arch_votes.get(key)
                if prev is None:
                    arch_votes[key] = (
                        int(label) if label is not None else None
                    )
        self.hypotheses = [
            DeckHypothesis(
                cards=cards,
                digest=_digest(list(cards)),
                prior_count=prior_count,
                archetype_index=arch_votes.get(cards),
            )
            for cards, prior_count in sorted(counts.items())
        ]
        self._native_digests = {hyp.digest for hyp in self.hypotheses}
        self._basic_prior = Counter(
            card_id
            for hyp in self.hypotheses
            for card_id in hyp.cards
            if card_id in _basic_pokemon_ids()
        )
        self.model_digest = _digest(
            {
                "schema": BELIEF_SCHEMA_VERSION,
                "sampler": self.sampler_name,
                "decks": [
                    (hyp.digest, hyp.prior_count, hyp.archetype_index)
                    for hyp in self.hypotheses
                ],
            }
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Optional[Path] = None,
        *,
        extra_deck_lists: Sequence[Sequence[int]] = (),
    ) -> "EmpiricalDeckPosterior":
        from . import archetypes
        from .baselines_runtime import ensure_baselines_installed, load_manifest
        from .deck_pool import read_deck

        specs = ensure_baselines_installed(load_manifest(manifest))
        # Discard baseline agent IDs/names (no opponent identity). Taxonomy
        # indices from card-signature classify_deck enable aux_head soft
        # reweight only — never used as a privileged opponent label.
        decks = [read_deck(spec.deck_csv) for spec in specs]
        arch_ids = archetypes.archetype_ids()
        indices: list[Optional[int]] = []
        for deck in decks:
            aid = archetypes.classify_deck(deck)
            indices.append(arch_ids.index(aid) if aid in arch_ids else None)
        for deck in extra_deck_lists:
            decks.append(list(deck))
            aid = archetypes.classify_deck(deck)
            indices.append(arch_ids.index(aid) if aid in arch_ids else None)
        return cls(decks, deck_archetype_indices=indices)

    def posterior(
        self,
        history: PublicBeliefHistory,
        *,
        neural: Optional[NeuralBeliefPriors] = None,
    ) -> list[tuple[DeckHypothesis, float]]:
        evidence = history.evidence_counter()
        total_evidence = sum(evidence.values())
        scored: list[tuple[DeckHypothesis, float]] = []
        for hyp in self.hypotheses:
            deck_count = Counter(hyp.cards)
            if any(deck_count[card] < count for card, count in evidence.items()):
                continue
            # Exact without-replacement evidence likelihood plus empirical prior.
            log_weight = math.log(float(hyp.prior_count))
            if total_evidence:
                log_weight -= math.log(math.comb(60, total_evidence))
                for card, count in evidence.items():
                    log_weight += math.log(math.comb(deck_count[card], count))
            scored.append((hyp, log_weight))
        if not scored:
            if total_evidence > 60:
                raise BeliefSupportError(
                    "public evidence exceeds a 60-card deck; particle support empty"
                )
            repaired: dict[tuple[int, ...], tuple[DeckHypothesis, float]] = {}
            for hyp in self.hypotheses:
                deck_count = Counter(hyp.cards)
                deficits = {
                    card: count - deck_count[card]
                    for card, count in evidence.items()
                    if count > deck_count[card]
                }
                repair_count = sum(deficits.values())
                for card, count in deficits.items():
                    deck_count[card] += count
                remove = repair_count
                for card in sorted(
                    deck_count,
                    key=lambda value: (
                        -(deck_count[value] - evidence.get(value, 0)),
                        value,
                    ),
                ):
                    surplus = max(
                        0, deck_count[card] - evidence.get(card, 0)
                    )
                    take = min(remove, surplus)
                    deck_count[card] -= take
                    remove -= take
                    if remove == 0:
                        break
                if remove:
                    continue
                cards = tuple(
                    card
                    for card, count in sorted(deck_count.items())
                    for _ in range(count)
                )
                if len(cards) != 60:
                    continue
                repaired_hyp = DeckHypothesis(
                    cards=cards,
                    digest=_digest(
                        {
                            "observable_history_repair": list(cards),
                            "source": hyp.digest,
                        }
                    ),
                    prior_count=hyp.prior_count,
                    archetype_index=hyp.archetype_index,
                )
                # Repairs are support recovery, not evidence that the repaired
                # synthetic deck was observed in the corpus.
                weight = math.log(float(hyp.prior_count)) - 4.0 * repair_count
                old = repaired.get(cards)
                if old is None or weight > old[1]:
                    repaired[cards] = (repaired_hyp, weight)
            scored = list(repaired.values())
            if not scored:
                raise BeliefSupportError(
                    "observable-history-consistent deck support is truly empty"
                )
        peak = max(weight for _, weight in scored)
        weights = [math.exp(weight - peak) for _, weight in scored]
        # Soft archetype prior from aux_head (distinct named head; optional).
        if (
            neural is not None
            and not neural.uniform_fallback
            and neural.archetype_logits is not None
        ):
            arch = _softmax(list(neural.archetype_logits))
            boosted: list[float] = []
            for (hyp, _), weight in zip(scored, weights):
                if hyp.archetype_index is None:
                    boosted.append(weight)
                elif 0 <= int(hyp.archetype_index) < len(arch):
                    boosted.append(weight * max(arch[int(hyp.archetype_index)], 1e-12))
                else:
                    boosted.append(weight)
            weights = boosted
        normalizer = sum(weights)
        return [
            (hyp, weight / normalizer)
            for (hyp, _), weight in zip(scored, weights)
        ]

    @property
    def config(self) -> dict[str, Any]:
        return {
            "schema_version": BELIEF_SCHEMA_VERSION,
            "sampler": self.sampler_name,
            "mode": "root_sampled_public_history_particles",
            "model_digest": self.model_digest,
            "hypotheses": len(self.hypotheses),
            "conserves_card_multiplicity": True,
            "uses_baseline_identity": False,
            "empty_support_policy": (
                "observable_history_conditioned_empirical_repair_or_fail"
            ),
            "setup_basic_conditioning": "public_legality_fact_only",
            "uses_privileged_state": False,
        }

    def sample_particle(
        self,
        obs: dict[str, Any],
        *,
        own_deck: Sequence[int],
        history: PublicBeliefHistory,
        rng: random.Random,
        neural: Optional[NeuralBeliefPriors] = None,
    ) -> HiddenStateParticle:
        current, seat = _state(obs)
        rows = self.posterior(history, neural=neural)
        own = tuple(int(card) for card in own_deck)
        if len(own) != 60:
            raise ValueError("own deck must have exactly 60 cards")
        use_nn = neural is not None and not neural.uniform_fallback
        own_inputs = self._sample_player_hidden(
            obs,
            player_index=seat,
            full_deck=own,
            include_hand=False,
            known_facedown_active=history.self_facedown_active_card,
            known_prizes_by_slot=dict(history.own_known_prizes),
            hand_logits=None,
            remainder_logits=None,
            rng=rng,
        )
        attempts = list(rows)
        errors: list[str] = []
        while attempts:
            weights = [row[1] for row in attempts]
            selected = rng.choices(range(len(attempts)), weights=weights, k=1)[0]
            hyp, probability = attempts.pop(selected)
            repair_count = 0 if hyp.digest in self._native_digests else 1
            try:
                conditioned, setup_repairs = self._condition_setup_basic(
                    obs,
                    player_index=1 - seat,
                    hypothesis=hyp,
                    evidence=history.evidence_counter(),
                    rng=rng,
                )
                repair_count += setup_repairs
                opp_inputs = self._sample_player_hidden(
                    obs,
                    player_index=1 - seat,
                    full_deck=conditioned.cards,
                    include_hand=True,
                    known_facedown_active=None,
                    known_prizes_by_slot=None,
                    hand_logits=(neural.opp_hand_logits if use_nn else None),
                    remainder_logits=(
                        neural.opp_remainder_logits if use_nn else None
                    ),
                    rng=rng,
                )
            except BeliefSupportError as exc:
                errors.append(str(exc))
                continue
            search_inputs = {
                "your_deck": own_inputs["deck"],
                "your_prize": own_inputs["prize"],
                "opponent_deck": opp_inputs["deck"],
                "opponent_prize": opp_inputs["prize"],
                "opponent_hand": opp_inputs["hand"],
                "opponent_active": opp_inputs["active"],
            }
            return HiddenStateParticle(
                search_inputs=search_inputs,
                opponent_deck=conditioned.cards,
                opponent_deck_digest=conditioned.digest,
                belief_model_digest=self.model_digest,
                public_history_digest=history.digest,
                posterior_probability=probability,
                support_mode=(
                    "empirical_exact"
                    if repair_count == 0
                    else "observable_history_conditioned_repair"
                ),
                support_repairs=repair_count,
            )
        detail = errors[0] if errors else "no posterior hypotheses"
        raise BeliefSupportError(
            "observable-history-consistent particle support is empty: " + detail
        )

    def _condition_setup_basic(
        self,
        obs: dict[str, Any],
        *,
        player_index: int,
        hypothesis: DeckHypothesis,
        evidence: Counter[int],
        rng: random.Random,
    ) -> tuple[DeckHypothesis, int]:
        """Condition an anonymous prior on the public fact of setup legality."""
        current, _seat = _state(obs)
        player = current["players"][player_index]
        active = list(player.get("active") or [])
        if not active or active[0] is not None:
            return hypothesis, 0
        consumed = _outside_prediction_counter(obs, player_index)
        consumed.update(_visible_prize_counter(player))
        deck_count = Counter(hypothesis.cards)
        available = deck_count - consumed
        if any(
            count > 0 and card in _basic_pokemon_ids()
            for card, count in available.items()
        ):
            return hypothesis, 0
        if not self._basic_prior:
            raise BeliefSupportError(
                "empirical prior has no Basic Pokemon for public setup fact"
            )
        removable = [
            card
            for card, count in available.items()
            if count > 0 and deck_count[card] > evidence.get(card, 0)
        ]
        if not removable:
            raise BeliefSupportError(
                "no unobserved card can be exchanged for setup Basic support"
            )
        basic_cards = list(self._basic_prior)
        basic = rng.choices(
            basic_cards,
            weights=[self._basic_prior[card] for card in basic_cards],
            k=1,
        )[0]
        removed = rng.choice(removable)
        deck_count[removed] -= 1
        deck_count[basic] += 1
        cards = tuple(
            card
            for card, count in sorted(deck_count.items())
            for _ in range(count)
        )
        repaired = DeckHypothesis(
            cards=cards,
            digest=_digest(
                {
                    "public_setup_basic_condition": list(cards),
                    "source": hypothesis.digest,
                }
            ),
            prior_count=hypothesis.prior_count,
            archetype_index=hypothesis.archetype_index,
        )
        return repaired, 1

    @staticmethod
    def _sample_player_hidden(
        obs: dict[str, Any],
        *,
        player_index: int,
        full_deck: Sequence[int],
        include_hand: bool,
        known_facedown_active: Optional[int],
        rng: random.Random,
        known_prizes_by_slot: Optional[dict[int, int]] = None,
        hand_logits: Optional[Sequence[float]] = None,
        remainder_logits: Optional[Sequence[float]] = None,
    ) -> dict[str, list[int]]:
        current, acting_seat = _state(obs)
        player = current["players"][player_index]
        outside = _outside_prediction_counter(obs, player_index)
        prizes = list(player.get("prize") or [])
        visible_prizes = _visible_prize_counter(player)
        remaining = Counter(int(card) for card in full_deck)
        remaining.subtract(outside)
        remaining.subtract(visible_prizes)
        known_prizes_by_slot = dict(known_prizes_by_slot or {})
        # Exact known own prizes are reserved out of the free pool first.
        for slot, card_id in sorted(known_prizes_by_slot.items()):
            if not (0 <= int(slot) < len(prizes)):
                raise BeliefSupportError(
                    f"known prize slot {slot} out of range for {len(prizes)} prizes"
                )
            if _card_id(prizes[int(slot)]) is not None:
                continue  # already accounted via visible_prizes
            remaining[int(card_id)] -= 1
        active_state = list(player.get("active") or [])
        if (
            not include_hand
            and active_state
            and active_state[0] is None
        ):
            if known_facedown_active is None:
                raise BeliefSupportError(
                    "own facedown active card is absent from self-private history"
                )
            remaining[known_facedown_active] -= 1
        if any(value < 0 for value in remaining.values()):
            raise BeliefSupportError(
                "deck hypothesis violates visible card multiplicities"
            )

        # When the acting player is selecting from a visible deck slice, libcg
        # ignores your_deck and keeps that serialized state. Exclude those cards
        # before assigning unknown prizes so one physical card is not duplicated.
        select = obs.get("select") or {}
        visible_select_deck = (
            select.get("deck")
            if player_index == acting_seat and select.get("deck") is not None
            else None
        )
        pool = [
            card_id
            for card_id, count in remaining.items()
            for _ in range(max(0, count))
        ]
        # Remainder prior reweights the private-pool multiset before zone fills.
        if remainder_logits is not None:
            rng.shuffle(pool)  # break ties before stable weighted draws
            pool = _weighted_sample_without_replacement(
                pool,
                len(pool),
                logit_lookup=remainder_logits,
                rng=rng,
            )
        else:
            rng.shuffle(pool)
        reserved_facedown_active: Optional[int] = None
        if include_hand and active_state and active_state[0] is None:
            eligible = [
                index
                for index, card_id in enumerate(pool)
                if card_id in _basic_pokemon_ids()
            ]
            if not eligible:
                raise BeliefSupportError(
                    "deck particle has no Basic Pokemon for facedown active"
                )
            # Condition all other hidden-zone draws on the public fact that a
            # legal Basic was placed active. Reserving it first avoids randomly
            # consuming every Basic into prizes/hand and spuriously rejecting a
            # valid deck hypothesis.
            reserved_facedown_active = pool.pop(rng.choice(eligible))
        prize_prediction: list[int] = []
        unknown_prize_slots = [
            slot
            for slot, prize in enumerate(prizes)
            if _card_id(prize) is None and slot not in known_prizes_by_slot
        ]
        # Fill unknown prizes then hand from remaining pool (hand uses hand logits).
        # Exact known prizes were already reserved out of ``remaining``/``pool``.
        if len(pool) < len(unknown_prize_slots):
            raise BeliefSupportError("not enough cards for prize particle")
        unknown_draws = pool[: len(unknown_prize_slots)]
        pool = pool[len(unknown_prize_slots) :]
        unknown_iter = iter(unknown_draws)
        for slot, prize in enumerate(prizes):
            visible = _card_id(prize)
            if visible is not None:
                prize_prediction.append(visible)
            elif slot in known_prizes_by_slot:
                prize_prediction.append(int(known_prizes_by_slot[slot]))
            else:
                prize_prediction.append(next(unknown_iter))
        hand: list[int] = []
        if include_hand:
            hand_count = int(player.get("handCount", 0))
            if hand_count > len(pool):
                raise BeliefSupportError("not enough cards for hand particle")
            hand = _weighted_sample_without_replacement(
                pool,
                hand_count,
                logit_lookup=hand_logits,
                rng=rng,
            )
            # Remove drawn hand cards from pool (multiset).
            pool_counter = Counter(pool)
            pool_counter.subtract(hand)
            if any(v < 0 for v in pool_counter.values()):
                raise BeliefSupportError("hand draw violated pool multiset")
            pool = [
                card_id
                for card_id, count in pool_counter.items()
                for _ in range(max(0, count))
            ]
            rng.shuffle(pool)
        active: list[int] = []
        if include_hand and active_state and active_state[0] is None:
            if reserved_facedown_active is None:
                raise BeliefSupportError("facedown active reservation was lost")
            active = [reserved_facedown_active]
        deck_count = int(player.get("deckCount", 0))
        if visible_select_deck:
            deck = []
            # libcg retains the serialized own deck at a visible deck-selection
            # decision and explicitly ignores this prediction field.
            pool = []
        else:
            if len(pool) != deck_count:
                raise BeliefSupportError("not enough cards for deck particle")
            deck = list(pool)
            pool = []
        if pool:
            zone_summary = {
                "player_index": player_index,
                "acting_seat": acting_seat,
                "outside": sum(outside.values()),
                "visible_prizes": sum(visible_prizes.values()),
                "deck_count": int(player.get("deckCount", 0)),
                "hand_count": int(player.get("handCount", 0)),
                "prize_slots": len(prizes),
                "known_own_prizes": len(known_prizes_by_slot),
                "active_facedown": bool(
                    list(player.get("active") or [])
                    and list(player.get("active") or [])[0] is None
                ),
                "looking": len(current.get("looking") or []),
                "select_deck": len((obs.get("select") or {}).get("deck") or []),
                "select_context": (obs.get("select") or {}).get("context"),
            }
            raise BeliefSupportError(
                f"particle card conservation mismatch: leftover={len(pool)} "
                f"zones={zone_summary}"
            )
        return {
            "deck": deck,
            "prize": prize_prediction,
            "hand": hand,
            "active": active,
        }


def simulator_version() -> str:
    """Digest the competition Python API and native simulator used by search."""
    # Submission tarballs vendor ``cg/`` beside ``main.py`` and expose that
    # location through CG_LIB_PATH.  Resolve the same runtime that search will
    # actually import instead of assuming the development-only kaggle/input
    # directory exists.
    runtime_dir = paths.cg_runtime_dir() / "cg"
    candidates = [
        runtime_dir / "api.py",
        runtime_dir / "sim.py",
        runtime_dir / "libcg.so",
    ]
    digest = hashlib.sha256()
    found = 0
    for path in candidates:
        if not path.is_file():
            continue
        found += 1
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError("competition simulator files unavailable")
    return f"competition-libcg-sha256:{digest.hexdigest()}"
