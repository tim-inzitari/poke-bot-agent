"""Public-state Alakazam SME adapter for the shadow tactical sequencer."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .alakazam_heuristics import (
    ABRA,
    ALAKAZAM,
    KADABRA,
    MIST_ENERGY,
    OPT_ATTACK,
    POWERFUL_HAND_ATTACK,
    guide_scores,
)
from .tactical_sequence_planner import RankedAction, TacticalSearchState


ALAKAZAM_TACTICAL_SME_SCHEMA = "poke_bot.alakazam_tactical_sme/v1"
class AlakazamTacticalSMEError(ValueError):
    """The public observation cannot support an Alakazam SME fact receipt."""


def _card_id(card: Any) -> int | None:
    if isinstance(card, Mapping):
        raw = card.get("id")
    else:
        raw = getattr(card, "id", None)
    try:
        return None if raw is None else int(raw)
    except (TypeError, ValueError):
        return None


def _zone(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _count(player: Mapping[str, Any], list_key: str, count_key: str) -> int:
    listed = player.get(list_key)
    if isinstance(listed, (list, tuple)):
        return len(listed)
    try:
        return max(0, int(player.get(count_key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _players(observation: Mapping[str, Any]) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise AlakazamTacticalSMEError("observation has no current state")
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise AlakazamTacticalSMEError("observation must expose exactly two players")
    try:
        actor = int(current.get("yourIndex", -1))
    except (TypeError, ValueError) as exc:
        raise AlakazamTacticalSMEError("yourIndex is invalid") from exc
    if actor not in (0, 1) or not all(isinstance(row, Mapping) for row in players):
        raise AlakazamTacticalSMEError("player/actor structure is invalid")
    return actor, players[actor], players[1 - actor]


def _remaining_hp(card: Any) -> int:
    if not isinstance(card, Mapping):
        return 0
    try:
        return max(0, int(card.get("hp", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _energy_ids(card: Any) -> tuple[int, ...]:
    if not isinstance(card, Mapping):
        return ()
    return tuple(
        card_id
        for card_id in (_card_id(row) for row in _zone(card.get("energyCards")))
        if card_id is not None
    )


def _board_counts(player: Mapping[str, Any]) -> Counter[int]:
    cards = _zone(player.get("active")) + _zone(player.get("bench"))
    return Counter(card_id for card_id in map(_card_id, cards) if card_id is not None)


@dataclass(frozen=True, slots=True)
class AlakazamPublicTacticalFacts:
    actor: int
    hand_cards: int
    deck_cards: int
    prizes_remaining: int
    opponent_active_remaining_hp: int
    powerful_hand_attack_legal: bool
    powerful_hand_required_hand_cards: int | None
    powerful_hand_hand_gap: int | None
    powerful_hand_current_lethal: bool
    mist_prevents_powerful_hand: bool
    replacement_line_live: bool
    visible_tutor_card_ids: tuple[int, ...]
    close_game_search_candidate: bool

    def to_public_facts(self) -> dict[str, Any]:
        return {
            "schema": ALAKAZAM_TACTICAL_SME_SCHEMA,
            "actor": self.actor,
            "hand_cards": self.hand_cards,
            "deck_cards": self.deck_cards,
            "prizes_remaining": self.prizes_remaining,
            "opponent_active_remaining_hp": self.opponent_active_remaining_hp,
            "powerful_hand_attack_legal": self.powerful_hand_attack_legal,
            "powerful_hand_required_hand_cards": (
                self.powerful_hand_required_hand_cards
            ),
            "powerful_hand_hand_gap": self.powerful_hand_hand_gap,
            "powerful_hand_current_lethal": self.powerful_hand_current_lethal,
            "mist_prevents_powerful_hand": self.mist_prevents_powerful_hand,
            "replacement_line_live": self.replacement_line_live,
            "visible_tutor_card_ids": list(self.visible_tutor_card_ids),
            "close_game_search_candidate": self.close_game_search_candidate,
            "source": "public_observation_and_training_only_alakazam_sme",
            "action_authority": False,
        }


def compile_alakazam_public_tactical_facts(
    observation: Mapping[str, Any],
) -> AlakazamPublicTacticalFacts:
    """Compile only public/acting-player-visible facts; never infer hidden cards."""

    actor, me, opponent = _players(observation)
    own_active = next(iter(_zone(me.get("active"))), None)
    opponent_active = next(iter(_zone(opponent.get("active"))), None)
    hand_cards = _count(me, "hand", "handCount")
    deck_cards = _count(me, "deck", "deckCount")
    prizes_remaining = _count(me, "prize", "prizeCount")
    remaining_hp = _remaining_hp(opponent_active)
    select = observation.get("select")
    if not isinstance(select, Mapping):
        select = {}
    options = _zone(select.get("option"))
    powerful_hand_legal = any(
        isinstance(option, Mapping)
        and int(option.get("type", -1) or -1) == OPT_ATTACK
        and int(option.get("attackId", -1) or -1) == POWERFUL_HAND_ATTACK
        for option in options
    )
    mist = MIST_ENERGY in _energy_ids(opponent_active)
    required = math.ceil(remaining_hp / 20) if remaining_hp > 0 else None
    gap = None if required is None else max(0, required - hand_cards)
    current_lethal = bool(
        _card_id(own_active) == ALAKAZAM
        and powerful_hand_legal
        and not mist
        and required is not None
        and hand_cards >= required
    )
    board = _board_counts(me)
    replacement_line_live = (
        board[ALAKAZAM] >= 2
        or (board[ALAKAZAM] >= 1 and (board[KADABRA] >= 1 or board[ABRA] >= 1))
    )
    visible_deck = select.get("deck")
    visible_tutor_cards = (
        tuple(
            card_id
            for card_id in (_card_id(card) for card in _zone(visible_deck))
            if card_id is not None
        )
        if visible_deck is not None
        else ()
    )
    # This is only a search trigger.  A simulator terminal receipt is still
    # required for any future lethal claim.
    close_candidate = bool(
        _card_id(own_active) == ALAKAZAM
        and powerful_hand_legal
        and not mist
        and required is not None
        and (current_lethal or prizes_remaining <= 2 or (gap is not None and gap <= 3))
    )
    return AlakazamPublicTacticalFacts(
        actor=actor,
        hand_cards=hand_cards,
        deck_cards=deck_cards,
        prizes_remaining=prizes_remaining,
        opponent_active_remaining_hp=remaining_hp,
        powerful_hand_attack_legal=powerful_hand_legal,
        powerful_hand_required_hand_cards=required,
        powerful_hand_hand_gap=gap,
        powerful_hand_current_lethal=current_lethal,
        mist_prevents_powerful_hand=mist,
        replacement_line_live=replacement_line_live,
        visible_tutor_card_ids=visible_tutor_cards,
        close_game_search_candidate=close_candidate,
    )


GuideScorer = Callable[..., Sequence[float] | None]


def rank_alakazam_sme_candidates(
    state: TacticalSearchState,
    policy_candidates: Sequence[RankedAction],
    *,
    deck: Iterable[int],
    score_actions: GuideScorer = guide_scores,
) -> tuple[RankedAction, ...]:
    """Keep r195's principal first; SME-rank only the discrepancy candidates."""

    candidates = tuple(policy_candidates)
    if not candidates:
        return ()
    ordered = tuple(sorted(candidates, key=lambda row: (-row.probability, row.action)))
    observation = state.raw_observation
    if not isinstance(observation, Mapping):
        return ordered
    scores = score_actions(
        dict(observation),
        [row.action for row in ordered],
        deck=tuple(int(card) for card in deck),
        force_enabled=True,
    )
    if scores is None or len(scores) != len(ordered):
        return ordered
    enriched = tuple(
        RankedAction(
            action=row.action,
            probability=row.probability,
            sme_priority=float(score),
            tactical_head_hint=row.tactical_head_hint,
        )
        for row, score in zip(ordered, scores)
    )
    principal = enriched[0]
    discrepancies = tuple(
        sorted(
            enriched[1:],
            key=lambda row: (-row.sme_priority, -row.probability, row.action),
        )
    )
    return (principal,) + discrepancies


__all__ = [
    "ALAKAZAM_TACTICAL_SME_SCHEMA",
    "AlakazamPublicTacticalFacts",
    "AlakazamTacticalSMEError",
    "compile_alakazam_public_tactical_facts",
    "rank_alakazam_sme_candidates",
]
