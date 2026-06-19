from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# cg.api LogType / AreaType numeric values (stable across training and submission).
LOG_TURN_START = 2
LOG_TURN_END = 3
LOG_DRAW = 4
LOG_DRAW_REVERSE = 5
LOG_MOVE_CARD = 6
LOG_MOVE_CARD_REVERSE = 7
LOG_PLAY = 10
LOG_ATTACH = 11

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_ENERGY = 8

DERIVED_INFERENCE_DIM = 16
DERIVED_FEATURE_NAMES = (
    "self_steps_since_last_draw",
    "self_draws_this_turn",
    "self_hand_avg_age",
    "self_hand_max_age",
    "self_cards_played_this_turn",
    "self_energy_discarded_since_last_draw",
    "self_supporter_cost_before_last_draw",
    "opp_steps_since_last_draw",
    "opp_draws_this_turn",
    "opp_hand_avg_age",
    "opp_hand_max_age",
    "opp_hand_min_age",
    "opp_cards_played_this_turn",
    "opp_energy_discarded_since_last_draw",
    "opp_hidden_hand_gains",
    "opp_visible_deck_to_hand",
)


def _norm_steps(steps: float) -> float:
    return float(np.tanh(max(0.0, steps) / 20.0))


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class _PlayerInferenceState:
    last_draw_step: int = 0
    draws_this_turn: int = 0
    cards_played_this_turn: int = 0
    energy_discarded_since_draw: int = 0
    hidden_hand_gains: int = 0
    visible_deck_to_hand: int = 0
    supporter_cost_before_last_draw: float = 0.0
    pending_supporter_play: bool = False


@dataclass
class GameEventTracker:
    """Track deduced draw/hand timing from incremental CABT logs."""

    step: int = 0
    turn: int = 0
    your_index: int = 0
    self_hand_serial_step: dict[int, int] = field(default_factory=dict)
    opp_hand_slot_steps: list[int] = field(default_factory=list)
    players: dict[int, _PlayerInferenceState] = field(
        default_factory=lambda: {0: _PlayerInferenceState(), 1: _PlayerInferenceState()}
    )

    def reset(self) -> None:
        self.step = 0
        self.turn = 0
        self.your_index = 0
        self.self_hand_serial_step.clear()
        self.opp_hand_slot_steps.clear()
        self.players = {0: _PlayerInferenceState(), 1: _PlayerInferenceState()}

    def observe(self, obs: dict[str, Any]) -> list[float]:
        current = obs.get("current") or {}
        self.step += 1
        self.turn = int(current.get("turn", self.turn) or 0)
        self.your_index = int(current.get("yourIndex", self.your_index) or 0)
        opponent_index = 1 - self.your_index

        for log in obs.get("logs") or []:
            self._process_log(log)

        self._sync_self_hand(current)
        self._sync_opponent_hand(current, opponent_index)
        return self.derived_features()

    def derived_features(self) -> list[float]:
        self_state = self.players[self.your_index]
        opp_state = self.players[1 - self.your_index]

        self_ages = [float(self.step - entered) for entered in self.self_hand_serial_step.values()]
        opp_ages = [float(self.step - entered) for entered in self.opp_hand_slot_steps]

        return [
            _norm_steps(float(self.step - self_state.last_draw_step)),
            float(self_state.draws_this_turn),
            _norm_steps(_avg(self_ages)),
            _norm_steps(max(self_ages) if self_ages else 0.0),
            float(self_state.cards_played_this_turn),
            float(self_state.energy_discarded_since_draw),
            float(self_state.supporter_cost_before_last_draw),
            _norm_steps(float(self.step - opp_state.last_draw_step)),
            float(opp_state.draws_this_turn),
            _norm_steps(_avg(opp_ages)),
            _norm_steps(max(opp_ages) if opp_ages else 0.0),
            _norm_steps(min(opp_ages) if opp_ages else 0.0),
            float(opp_state.cards_played_this_turn),
            float(opp_state.energy_discarded_since_draw),
            float(opp_state.hidden_hand_gains),
            float(opp_state.visible_deck_to_hand),
        ]

    def _player(self, player_index: int) -> _PlayerInferenceState:
        return self.players[int(player_index)]

    def _record_draw(self, player_index: int, *, serial: int | None, supporter_cost: bool) -> None:
        state = self._player(player_index)
        state.last_draw_step = self.step
        state.draws_this_turn += 1
        state.energy_discarded_since_draw = 0
        state.supporter_cost_before_last_draw = 1.0 if supporter_cost else 0.0
        state.pending_supporter_play = False

        if player_index == self.your_index and serial is not None:
            self.self_hand_serial_step[serial] = self.step
        elif player_index != self.your_index:
            self.opp_hand_slot_steps.append(self.step)

    def _record_hand_entry(self, player_index: int, *, serial: int | None) -> None:
        if player_index == self.your_index and serial is not None:
            self.self_hand_serial_step.setdefault(serial, self.step)
        elif player_index != self.your_index:
            self.opp_hand_slot_steps.append(self.step)

    def _record_play_from_hand(self, player_index: int, *, serial: int | None) -> None:
        state = self._player(player_index)
        state.cards_played_this_turn += 1
        if player_index == self.your_index:
            if serial is not None:
                self.self_hand_serial_step.pop(serial, None)
            elif self.self_hand_serial_step:
                oldest_serial = min(self.self_hand_serial_step, key=self.self_hand_serial_step.get)
                self.self_hand_serial_step.pop(oldest_serial, None)
        elif self.opp_hand_slot_steps:
            self.opp_hand_slot_steps.pop(0)

    def _record_energy_discard(self, player_index: int) -> None:
        self._player(player_index).energy_discarded_since_draw += 1

    def _process_log(self, log: dict[str, Any]) -> None:
        log_type = int(log.get("type", -1))
        player_index = int(log.get("playerIndex", -1))
        if player_index not in (0, 1):
            return

        if log_type == LOG_TURN_START:
            state = self._player(player_index)
            state.draws_this_turn = 0
            state.cards_played_this_turn = 0
            state.hidden_hand_gains = 0
            state.visible_deck_to_hand = 0
            return

        if log_type == LOG_TURN_END:
            return

        if log_type == LOG_DRAW:
            self._record_draw(
                player_index,
                serial=self._optional_int(log.get("serial")),
                supporter_cost=self._player(player_index).pending_supporter_play,
            )
            return

        if log_type == LOG_DRAW_REVERSE:
            self._record_draw(
                player_index,
                serial=None,
                supporter_cost=self._player(player_index).pending_supporter_play,
            )
            return

        if log_type in {LOG_MOVE_CARD, LOG_MOVE_CARD_REVERSE}:
            from_area = int(log.get("fromArea", 0) or 0)
            to_area = int(log.get("toArea", 0) or 0)
            serial = self._optional_int(log.get("serial"))

            if from_area == AREA_DECK and to_area == AREA_HAND:
                self._record_hand_entry(player_index, serial=serial)
                if player_index != self.your_index and log_type == LOG_MOVE_CARD:
                    self._player(player_index).visible_deck_to_hand += 1
                return

            if to_area == AREA_DISCARD and from_area in {AREA_HAND, AREA_ENERGY, AREA_ACTIVE}:
                if from_area in {AREA_ENERGY, AREA_ACTIVE}:
                    self._record_energy_discard(player_index)
                if from_area == AREA_HAND:
                    self._record_play_from_hand(player_index, serial=serial)
                return

            if from_area == AREA_HAND and to_area in {AREA_ACTIVE, AREA_BENCH}:
                self._record_play_from_hand(player_index, serial=serial)
            return

        if log_type == LOG_PLAY:
            self._player(player_index).pending_supporter_play = True
            self._record_play_from_hand(
                player_index,
                serial=self._optional_int(log.get("serial")),
            )
            return

        if log_type == LOG_ATTACH:
            # Energy attachment from hand costs a card; treat as leaving hand.
            self._record_play_from_hand(
                player_index,
                serial=self._optional_int(log.get("serial")),
            )

    def _sync_self_hand(self, current: dict[str, Any]) -> None:
        players = current.get("players") or [{}, {}]
        if self.your_index >= len(players):
            return
        hand = players[self.your_index].get("hand")
        if hand is None:
            return

        visible_serials = {
            int(card["serial"])
            for card in hand
            if isinstance(card, dict) and card.get("serial") is not None
        }
        for serial in visible_serials:
            self.self_hand_serial_step.setdefault(serial, self.step)

        for serial in list(self.self_hand_serial_step):
            if serial not in visible_serials:
                self.self_hand_serial_step.pop(serial, None)

    def _sync_opponent_hand(self, current: dict[str, Any], opponent_index: int) -> None:
        players = current.get("players") or [{}, {}]
        if opponent_index >= len(players):
            return
        hand_count = int(players[opponent_index].get("handCount", 0) or 0)

        while len(self.opp_hand_slot_steps) < hand_count:
            self.opp_hand_slot_steps.append(self.step)
            self._player(opponent_index).hidden_hand_gains += 1

        while len(self.opp_hand_slot_steps) > hand_count:
            self.opp_hand_slot_steps.pop(0)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)
