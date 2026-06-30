from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class CardId:
    DWEBBLE = 344
    CRUSTLE = 345

    MAKUHITA = 673
    HARIYAMA = 674
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    DREEPY = 119
    DRAKLOAK = 120
    DRAGAPULT_EX = 121

    ABRA = 741
    KADABRA = 742
    ALAKAZAM = 743
    DUNSPARCE = 305
    DUDUNSPARCE = 66

    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    BUDDY_BUDDY_POFFIN = 1086
    RARE_CANDY = 1079
    POKE_PAD = 1152
    CRUSHING_HAMMER = 1120
    ENHANCED_HAMMER = 1081
    LILLIE_DETERMINATION = 1227
    CARMINE = 1192
    CRISPIN = 1198

    BASIC_FIGHTING = 6
    BASIC_PSYCHIC = 5
    BASIC_FIRE = 2

    MIST_ENERGY = 11
    ROCK_FIGHTING_ENERGY = 20
    LEGACY_ENERGY = 12


ATTACK_PRIORITY_IDS = {
    # Public baselines commonly key on these exact attacks.
    983,   # Mega Lucario ex: Mega Brave
    1072,  # Alakazam: Powerful Hand
}


@dataclass(frozen=True)
class AttackPlan:
    attacker_index: int = -1
    target_index: int = -1
    attack_id: int | None = None
    target_score: float = 0.0
    needs_energy: bool = False
    can_win_game: bool = False


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "value"):
        return value.value
    return value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def card_id(card: Any) -> int | None:
    raw = _get(card, "id", _get(card, "cardId"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def hp(card: Any) -> int:
    raw = _get(card, "hp", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def max_hp(card: Any) -> int:
    raw = _get(card, "maxHp", _get(card, "max_hp", hp(card)))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return hp(card)


def energy_count(pokemon: Any) -> int:
    energies = _get(pokemon, "energies", None)
    if energies is not None:
        return len(_items(energies))
    return len(_items(_get(pokemon, "energyCards", [])))


def tool_count(pokemon: Any) -> int:
    return len(_items(_get(pokemon, "tools", [])))


def prize_count(pokemon: Any) -> int:
    cid = card_id(pokemon)
    if cid is None:
        return 1
    # A small static prior is enough for action ranking; exact card metadata is not
    # always available in stripped submission/runtime paths.
    if cid in {CardId.MEGA_LUCARIO_EX}:
        count = 3
    elif str(_get(pokemon, "ex", "")).lower() == "true" or cid in {CardId.DRAGAPULT_EX}:
        count = 2
    else:
        count = 1
    for energy in _items(_get(pokemon, "energyCards", [])):
        if card_id(energy) == CardId.LEGACY_ENERGY:
            count -= 1
    return max(1, count)


def get_player(obs: Any, player_index: int) -> Any | None:
    current = _get(obs, "current")
    players = _items(_get(current, "players", []))
    if 0 <= player_index < len(players):
        return players[player_index]
    return None


def current_player_index(obs: Any) -> int:
    try:
        return int(_get(_get(obs, "current"), "yourIndex", 0))
    except (TypeError, ValueError):
        return 0


def get_card(obs: Any, area: Any, index: int, player_index: int) -> Any | None:
    area_name = _enum_name(area)
    player = get_player(obs, player_index)
    try:
        if area_name == "DECK":
            return _items(_get(_get(obs, "select"), "deck", []))[index]
        if area_name == "HAND":
            return _items(_get(player, "hand", []))[index]
        if area_name == "DISCARD":
            return _items(_get(player, "discard", []))[index]
        if area_name == "ACTIVE":
            return _items(_get(player, "active", []))[index]
        if area_name == "BENCH":
            return _items(_get(player, "bench", []))[index]
        if area_name == "PRIZE":
            return _items(_get(player, "prize", []))[index]
        if area_name == "STADIUM":
            return _items(_get(_get(obs, "current"), "stadium", []))[index]
        if area_name == "LOOKING":
            return _items(_get(_get(obs, "current"), "looking", []))[index]
    except (IndexError, TypeError):
        return None
    return None


def board(obs: Any, player_index: int) -> list[Any]:
    player = get_player(obs, player_index)
    return [p for p in _items(_get(player, "active", [])) + _items(_get(player, "bench", [])) if p is not None]


def opponent_has(obs: Any, ids: set[int]) -> bool:
    my_index = current_player_index(obs)
    return any(card_id(pokemon) in ids for pokemon in board(obs, 1 - my_index))


def low_deck(obs: Any, threshold: int = 8) -> bool:
    player = get_player(obs, current_player_index(obs))
    try:
        return int(_get(player, "deckCount", 99)) <= threshold
    except (TypeError, ValueError):
        return False


def target_score(pokemon: Any) -> float:
    cid = card_id(pokemon) or -1
    score = prize_count(pokemon) * 1000.0
    score += energy_count(pokemon) * 150.0
    score += tool_count(pokemon) * 100.0
    score += min(hp(pokemon), 340)
    if cid in {CardId.RIOLU, CardId.DREEPY, CardId.ABRA}:
        score += 350.0
    if cid in {CardId.MEGA_LUCARIO_EX, CardId.DRAGAPULT_EX, CardId.ALAKAZAM}:
        score += 250.0
    if cid in {CardId.CRUSTLE}:
        score += 500.0
    return score


def estimate_attack_damage(option: Any, attacker: Any | None = None) -> int:
    attack_id = _get(option, "attackId")
    try:
        attack_id = int(_value(attack_id))
    except (TypeError, ValueError):
        attack_id = None
    attacker_id = card_id(attacker)
    if attack_id == 983:
        return 270
    if attack_id == 1072:
        return 160
    if attacker_id == CardId.DRAGAPULT_EX:
        return 200
    if attacker_id == CardId.HARIYAMA:
        return 210
    if attacker_id == CardId.MEGA_LUCARIO_EX:
        return 130
    if attacker_id == CardId.CRUSTLE:
        return 150
    return 120


def build_attack_plan(obs: Any) -> AttackPlan:
    select = _get(obs, "select")
    if select is None or _enum_name(_get(select, "context")) != "MAIN":
        return AttackPlan()

    options = _items(_get(select, "option", []))
    my_index = current_player_index(obs)
    opponent_index = 1 - my_index
    my_board = board(obs, my_index)
    op_board = board(obs, opponent_index)
    if not op_board:
        return AttackPlan()

    active = my_board[0] if my_board else None
    active_id = card_id(active)
    crustle_wall = opponent_has(obs, {CardId.DWEBBLE, CardId.CRUSTLE})
    best = AttackPlan()
    best_score = -math.inf
    can_gust = any(
        _enum_name(_get(option, "type")) == "PLAY"
        and card_id(get_card(obs, "HAND", int(_get(option, "index", 0)), my_index)) == CardId.BOSS_ORDERS
        for option in options
    )

    attack_options = [option for option in options if _enum_name(_get(option, "type")) == "ATTACK"]
    if not attack_options:
        return AttackPlan()

    for attack_option in attack_options:
        damage = estimate_attack_damage(attack_option, active)
        for target_index, target in enumerate(op_board):
            if target_index > 0 and not can_gust:
                continue
            if crustle_wall and active_id == CardId.MEGA_LUCARIO_EX and card_id(target) == CardId.CRUSTLE:
                continue
            score = target_score(target)
            remaining = hp(target) - damage
            if remaining <= 0:
                score += 5000.0 + prize_count(target) * 1500.0
            else:
                score *= max(0.2, min(1.0, damage / max(1, hp(target))))
            if target_index == 0:
                score += 350.0
            attack_id = _get(attack_option, "attackId")
            try:
                attack_id = int(_value(attack_id))
            except (TypeError, ValueError):
                attack_id = None
            if attack_id in ATTACK_PRIORITY_IDS:
                score += 250.0
            my_player = get_player(obs, my_index)
            try:
                # In Pokemon TCG you win by taking all of YOUR remaining Prize
                # cards. A KO matters as a wincon when its prize value covers
                # our remaining prize list, not the opponent's.
                can_win = remaining <= 0 and len(_items(_get(my_player, "prize", []))) <= prize_count(target)
            except TypeError:
                can_win = False
            if can_win:
                score += 50_000.0
            if score > best_score:
                best_score = score
                best = AttackPlan(
                    attacker_index=0,
                    target_index=target_index,
                    attack_id=attack_id,
                    target_score=score,
                    can_win_game=can_win,
                )
    return best


def score_option(obs: Any, option: Any, plan: AttackPlan | None = None) -> float:
    select = _get(obs, "select")
    context = _enum_name(_get(select, "context"))
    option_type = _enum_name(_get(option, "type"))
    my_index = current_player_index(obs)
    plan = plan or build_attack_plan(obs)

    if option_type == "NUMBER":
        return float(_get(option, "number", 0) or 0)
    if option_type == "YES":
        return 100.0 if context == "IS_FIRST" else 5.0
    if option_type == "NO":
        return 0.0

    if option_type == "CARD":
        card = get_card(obs, _get(option, "area"), int(_get(option, "index", 0)), int(_get(option, "playerIndex", my_index)))
        if card is None:
            return 0.0
        if int(_get(option, "playerIndex", my_index)) != my_index:
            return target_score(card) + (500.0 if _enum_name(_get(option, "area")) == "ACTIVE" else 0.0)
        cid = card_id(card)
        score = hp(card) + energy_count(card) * 75.0
        if context in {"TO_HAND", "EVOLVE", "TO_BENCH"}:
            if cid in {
                CardId.RIOLU,
                CardId.MEGA_LUCARIO_EX,
                CardId.MAKUHITA,
                CardId.HARIYAMA,
                CardId.DREEPY,
                CardId.DRAKLOAK,
                CardId.DRAGAPULT_EX,
                CardId.ABRA,
                CardId.KADABRA,
                CardId.ALAKAZAM,
                CardId.CRUSTLE,
            }:
                score += 1000.0
        return score

    if context != "MAIN":
        return 10.0

    if option_type == "ABILITY":
        return -10.0 if low_deck(obs) else 12_000.0
    if option_type == "EVOLVE":
        return 9_000.0
    if option_type == "ATTACH":
        card = get_card(obs, "HAND", int(_get(option, "index", 0)), my_index)
        target = get_card(obs, _get(option, "inPlayArea"), int(_get(option, "inPlayIndex", 0)), my_index)
        score = 7_000.0
        if card_id(card) == CardId.HERO_CAPE:
            score += 1500.0 if _enum_name(_get(option, "inPlayArea")) == "ACTIVE" else 250.0
        if target is not None:
            score += max(0, 3 - energy_count(target)) * 200.0
            if card_id(target) in {CardId.HARIYAMA, CardId.MEGA_LUCARIO_EX, CardId.DRAGAPULT_EX, CardId.ALAKAZAM}:
                score += 500.0
        return score
    if option_type == "PLAY":
        card = get_card(obs, "HAND", int(_get(option, "index", 0)), my_index)
        cid = card_id(card)
        if cid in {CardId.BUDDY_BUDDY_POFFIN, CardId.POKE_PAD, CardId.LILLIE_DETERMINATION, CardId.CARMINE} and low_deck(obs):
            return -1.0
        if cid == CardId.BOSS_ORDERS:
            return 8_000.0 if plan.target_index > 0 else -1.0
        if cid in {CardId.RARE_CANDY, CardId.BUDDY_BUDDY_POFFIN, CardId.POKE_PAD}:
            return 10_500.0
        if cid in {CardId.CRUSHING_HAMMER, CardId.ENHANCED_HAMMER}:
            return 4_000.0
        return 6_000.0
    if option_type == "RETREAT":
        return 3_000.0 if plan.attacker_index > 0 else -1.0
    if option_type == "ATTACK":
        attack_id = _get(option, "attackId")
        try:
            attack_id = int(_value(attack_id))
        except (TypeError, ValueError):
            attack_id = None
        score = 1_000.0 + plan.target_score * 0.1
        if attack_id == plan.attack_id:
            score += 2_000.0
        if plan.can_win_game:
            score += 20_000.0
        return score
    return 0.0


def score_actions_with_attack_plan(obs_dict: dict[str, Any], actions: list[list[int]]) -> list[float]:
    """Return action priors learned from public rule agents' attack-plan pattern."""
    try:
        from cg.api import to_observation_class

        obs = to_observation_class(obs_dict)
    except Exception:
        return [0.0 for _ in actions]

    select = _get(obs, "select")
    options = _items(_get(select, "option", []))
    plan = build_attack_plan(obs)
    option_scores = [score_option(obs, option, plan) for option in options]
    action_scores: list[float] = []
    for action in actions:
        if not action:
            action_scores.append(0.0)
            continue
        total = 0.0
        for order, option_index in enumerate(action):
            if 0 <= option_index < len(option_scores):
                total += option_scores[option_index] - order * 0.01
        action_scores.append(total / max(1, len(action)))
    return action_scores


def choose_attack_plan_action(obs_dict: dict[str, Any], actions: list[list[int]]) -> list[int]:
    if not actions:
        return []
    scores = score_actions_with_attack_plan(obs_dict, actions)
    return max(zip(scores, actions), key=lambda item: item[0])[1]
