
from __future__ import annotations

import os
import random
from collections import defaultdict
import time

try:
    from cg.api import SearchState, search_begin, search_step, search_end
    _SEARCH_AVAILABLE = True
except Exception:
    _SEARCH_AVAILABLE = False

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    STONJOURNER = 682
    # NOTE: Rolycoly card_id is estimated. Verify on Kaggle with:
    #   [c for c in all_card_data() if c.name == "Rolycoly"]
    ROLYCOLY = 704

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


# Track Boundless Power cooldown (Stonjourner's 2nd attack)
boundless_power_turn = -1
LOW_DECK_COUNT = 8


def _resolve_deck_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(base_dir, "deck.csv"),
        "deck.csv",
        "/kaggle_simulations/agent/deck.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("deck.csv not found in: " + ", ".join(candidates))


DECK_PATH = _resolve_deck_path()
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]
if len(my_deck) != 60:
    raise ValueError(f"deck.csv must contain 60 card ids, got {len(my_deck)}")


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False

# Optional search hook. The Phase 1 notebook keeps this disabled; it exists so later
# experiments can be connected without changing the submission contract.
USE_SEARCH = True

# --- Phase-1 diagnostics: distinguish "policy ran" from "fell back". ---------
_DIAG = {"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
         "obs_fallback": 0, "deck_returns": 0, "errors": {},
         "search_used": 0, "search_failed": 0,
         "chosen_types": {}, "attack_ids_chosen": {},
         "attack_opts_by_active": {}, "boundless_power_used": 0}


def _diag_record_error(exc):
    key = type(exc).__name__ + ": " + str(exc)[:160]
    _DIAG["errors"][key] = _DIAG["errors"].get(key, 0) + 1


def diag_reset():
    _DIAG.update({"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
                  "obs_fallback": 0, "deck_returns": 0, "errors": {},
                  "search_used": 0, "search_failed": 0,
                  "chosen_types": {}, "attack_ids_chosen": {},
                  "attack_opts_by_active": {}, "boundless_power_used": 0})


def diag_snapshot():
    snap = {}
    for k, v in _DIAG.items():
        if k == "attack_opts_by_active":
            snap[k] = {kk: sorted(vv) for kk, vv in v.items()}
        elif isinstance(v, dict):
            snap[k] = dict(v)
        else:
            snap[k] = v
    dec = max(1, snap.get("decisions", 0))
    snap["fallback_rate"] = (snap.get("policy_fallback", 0) + snap.get("obs_fallback", 0)) / dec
    sd = snap.get("search_used", 0) + snap.get("search_failed", 0)
    snap["search_fail_rate"] = (snap.get("search_failed", 0) / sd) if sd else 0.0
    return snap


def _diag_observe(obs):
    """Record, on the REAL obs, which attacks are legal (reveals true attackIds)."""
    try:
        sel = obs.select
        st = obs.current
        if sel is None or st is None:
            return
        me = st.players[st.yourIndex]
        active_id = me.active[0].id if (me.active and me.active[0] is not None) else -1
        for opt in sel.option:
            if opt.type == OptionType.ATTACK:
                aid = getattr(opt, "attackId", None)
                if aid is not None:
                    _DIAG["attack_opts_by_active"].setdefault(active_id, set()).add(aid)
    except Exception:
        pass


def _diag_observe_choice(obs, selection):
    """Record which option TYPES (and attackIds) actually get chosen."""
    try:
        sel = obs.select
        if sel is None or not selection:
            return
        for i in selection:
            if 0 <= i < len(sel.option):
                opt = sel.option[i]
                key = str(opt.type)
                _DIAG["chosen_types"][key] = _DIAG["chosen_types"].get(key, 0) + 1
                if opt.type == OptionType.ATTACK:
                    aid = getattr(opt, "attackId", None)
                    if aid is not None:
                        _DIAG["attack_ids_chosen"][aid] = _DIAG["attack_ids_chosen"].get(aid, 0) + 1
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Forward-search over the engine model (idea 1). Strictly additive & safe: it
# only re-ranks among the legal MAIN options, resolves sub-selects with the
# greedy policy inside simulation, evaluates board value at the end of MY turn,
# and on ANY engine deviation returns None -> greedy (counted as search_failed,
# which does NOT inflate fallback_rate). Toggle with USE_SEARCH.
# ---------------------------------------------------------------------------
SEARCH_NODE_BUDGET = 160        # max forward-steps per decision
SEARCH_TIME_BUDGET_S = 1.8      # wall-clock cap per decision (respect 10-min/game)
SEARCH_OPPONENT_PLY = False     # 1b (opponent reply) -- experimental, off by default


def _board_value(obs, my_index):
    """Win-aligned leaf score; prize differential dominates."""
    st = obs.current
    res = getattr(st, "result", -1)
    if res is not None and res >= 0:
        if res == my_index:
            return 1e6
        if res == (1 - my_index):
            return -1e6
        return 0.0
    me = st.players[my_index]
    op = st.players[1 - my_index]
    val = (len(op.prize) - len(me.prize)) * 1000.0
    op_active = op.active[0] if op.active else None
    if op_active is not None:
        data = card_table.get(op_active.id)
        maxhp = data.hp if data is not None else op_active.hp
        try:
            val += max(0, maxhp - op_active.hp) * 2.0
        except Exception:
            pass
    my_active = me.active[0] if me.active else None
    if my_active is not None:
        data = card_table.get(my_active.id)
        maxhp = data.hp if data is not None else my_active.hp
        try:
            val -= max(0, maxhp - my_active.hp) * 1.5
        except Exception:
            pass
        # Stonjourner on active is extremely valuable vs ex opponents
        if my_active.id == C.STONJOURNER and op_active is not None:
            op_data = card_table.get(op_active.id)
            if op_data is not None and (op_data.ex or op_data.megaEx):
                val += 500  # immune to ex attacks - huge defensive bonus
    for p in (me.active + me.bench):
        if p is not None:
            val += len(p.energies) * 4.0
            # Stonjourner on field is valuable (prize liability = 1, not 2-3)
            if p.id == C.STONJOURNER:
                val += 40.0
    bench_count = sum(1 for p in me.bench if p is not None)
    val += bench_count * 30.0
    op_bench_count = sum(1 for p in op.bench if p is not None)
    val -= op_bench_count * 20.0
    val += me.handCount * 8.0
    val -= op.handCount * 8.0
    if me.deckCount <= 5:
        val -= (6 - me.deckCount) * 50.0
    return val


def _enumerate_actions(select, cap=20):
    n = len(select.option)
    k = select.maxCount
    if n == 0 or k == 0:
        return []
    if k == 1:
        return [[i] for i in range(min(n, cap))]
    out = []
    idx = list(range(k))
    for _ in range(cap):
        out.append(idx.copy())
        for i in range(k):
            j = k - i - 1
            if idx[j] < n - i - 1:
                idx[j] += 1
                for m in range(j + 1, k):
                    idx[m] = idx[m - 1] + 1
                break
        else:
            break
    return out


def _greedy_select_sim(obs):
    try:
        return StonjournerPolicy(obs).choose()
    except Exception:
        return _legal_fallback(obs.select)


def _search_value(state, my_index, budget):
    obs = state.observation
    st = obs.current
    res = getattr(st, "result", -1)
    if res is not None and res >= 0:
        return _board_value(obs, my_index)
    if st.yourIndex != my_index:
        return _board_value(obs, my_index)     # opponent to move -> leaf (1a)
    sel = obs.select
    if sel is None or len(sel.option) == 0:
        return _board_value(obs, my_index)
    over = budget["n"] >= SEARCH_NODE_BUDGET or (time.time() - budget["t0"]) > SEARCH_TIME_BUDGET_S
    if sel.context == SelectContext.MAIN and not over:
        best = None
        for a in _enumerate_actions(sel):
            if budget["n"] >= SEARCH_NODE_BUDGET:
                break
            budget["n"] += 1
            try:
                nxt = search_step(state.searchId, a)
            except Exception:
                continue
            v = _search_value(nxt, my_index, budget)
            if best is None or v > best:
                best = v
        return best if best is not None else _board_value(obs, my_index)
    budget["n"] += 1
    try:
        nxt = search_step(state.searchId, _greedy_select_sim(obs))
    except Exception:
        return _board_value(obs, my_index)
    return _search_value(nxt, my_index, budget)


def search_plan(obs_dict, obs, ranked, scores):
    if not _SEARCH_AVAILABLE:
        return None
    try:
        if obs.select is None or obs.current is None:
            return None
        if obs.select.context != SelectContext.MAIN:
            return None
        if obs.select.maxCount == 0 or len(obs.select.option) <= 1:
            return None
        my_index = obs.current.yourIndex
        st = obs.current
        me = st.players[my_index]
        opp = st.players[1 - my_index]
        opp_active = opp.active

        def _samp(pool, k):
            if k <= 0:
                return []
            if k <= len(pool):
                return random.sample(pool, k)
            return (pool * (k // max(1, len(pool)) + 1))[:k]

        state = search_begin(
            obs,
            your_deck=_samp(my_deck, me.deckCount),
            your_prize=_samp(my_deck, len(me.prize)),
            opponent_deck=_samp(my_deck, opp.deckCount),    # mirror-plausible fill
            opponent_prize=[1] * len(opp.prize),
            opponent_hand=[1] * opp.handCount,
            opponent_active=[1072] if (len(opp_active) > 0 and opp_active[0] is None) else [],
        )
        budget = {"n": 0, "t0": time.time()}
        best_sel, best_val = None, None
        root = state.observation
        for action in _enumerate_actions(root.select):
            if budget["n"] >= SEARCH_NODE_BUDGET or (time.time() - budget["t0"]) > SEARCH_TIME_BUDGET_S:
                break
            budget["n"] += 1
            try:
                child = search_step(state.searchId, action)
            except Exception:
                continue
            v = _search_value(child, my_index, budget)
            if best_val is None or v > best_val:
                best_val, best_sel = v, action
        try:
            search_end()
        except Exception:
            pass
        if best_sel is None:
            _DIAG["search_failed"] += 1
            return None
        _DIAG["search_used"] += 1
        return best_sel
    except Exception as exc:
        _diag_record_error(exc)
        _DIAG["search_failed"] += 1
        try:
            search_end()
        except Exception:
            pass
        return None


def normalize_selection(ranked, scores, select):
    n = len(select.option)
    minc = max(0, min(select.minCount, n))
    maxc = max(minc, min(select.maxCount, n))

    out = []
    seen = set()
    for i in ranked:
        if not (0 <= i < n) or i in seen:
            continue
        score = scores[i] if i < len(scores) else 0
        # Score 0 usually means an unmodelled neutral option. For optional
        # selections, do not take it unless minCount forces a choice.
        if score > 0 or len(out) < minc:
            out.append(i)
            seen.add(i)
        if len(out) >= maxc:
            break

    for i in range(n):
        if len(out) >= minc:
            break
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _legal_fallback(select):
    try:
        n = len(select.option)
        k = min(max(0, select.minCount), n)
        return list(range(k))
    except Exception:
        return []


def _legal_fallback_from_dict(obs_dict):
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        minc = sel.get("minCount", 0)
        n = len(opts)
        k = min(max(0, minc), n)
        return list(range(k))
    except Exception:
        return []

def _safe_get(seq, index: int):
    try:
        if seq is None or index is None or index < 0 or index >= len(seq):
            return None
        return seq[index]
    except Exception:
        return None


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    try:
        player = obs.current.players[player_index]
        match area:
            case AreaType.DECK:
                return _safe_get(getattr(obs.select, "deck", None), index)
            case AreaType.HAND:
                return _safe_get(getattr(player, "hand", None), index)
            case AreaType.DISCARD:
                return _safe_get(getattr(player, "discard", None), index)
            case AreaType.ACTIVE:
                return _safe_get(getattr(player, "active", None), index)
            case AreaType.BENCH:
                return _safe_get(getattr(player, "bench", None), index)
            case AreaType.PRIZE:
                return _safe_get(getattr(player, "prize", None), index)
            case AreaType.STADIUM:
                return _safe_get(getattr(obs.current, "stadium", None), index)
            case AreaType.LOOKING:
                return _safe_get(getattr(obs.current, "looking", None), index)
            case _:
                return None
    except Exception:
        return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return prize_count(pokemon) * 1000 + getattr(pokemon, "hp", 0)
    score = prize_count(pokemon) * 1000
    # Ex Pokemon are high-value targets (2-3 prizes, and Stonjourner can safely attack them)
    if data.ex or data.megaEx:
        score += 300
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:  # low-value support Pokemon
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    try:
        maxhp = data.hp if data else getattr(pokemon, "hp", 100)
        if maxhp > 0 and pokemon.hp > 0:
            hp_ratio = pokemon.hp / maxhp
            if hp_ratio <= 0.3:
                score += 500
            elif hp_ratio <= 0.5:
                score += 200
    except Exception:
        pass
    return score


class StonjournerPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_stonjourner = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self._attack_option_map = {}  # option_index -> attack_index
        self._attack_ids = {}  # attack_index -> attackId

        self._count_cards()
        self._scan_main_options()

    def rank(self) -> tuple[list[int], list[float]]:
        if not self.select.option or self.select.maxCount == 0:
            return [], []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked, scores

    def choose(self) -> list[int]:
        ranked, scores = self.rank()
        selection = normalize_selection(ranked, scores, self.select)
        self.remember_chosen_side_effects(selection)
        return selection

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id == C.STONJOURNER and len(pokemon.energies) >= 1:
                self.has_ready_stonjourner = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        atk_count = 0
        for i, option in enumerate(self.select.option):
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is None:
                    continue
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                self._attack_option_map[i] = atk_count
                aid = getattr(option, "attackId", -1)
                self._attack_ids[atk_count] = aid
                atk_count += 1

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _is_boundless_power_on_cooldown(self) -> bool:
        """Check if Stonjourner used Boundless Power last turn."""
        global boundless_power_turn
        if boundless_power_turn >= 0 and boundless_power_turn == self.state.turn - 1:
            return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.STONJOURNER:
            if attack_index == 0:
                # Stony Kick: 1F energy, 20 damage + 20 to a benched Pokemon
                energy_required = 1
                base_damage = 20
                base_score = 10  # chip damage + bench damage
            elif attack_index == 1:
                # Boundless Power: 3F energy, 140 damage, can't attack next turn
                if self._is_boundless_power_on_cooldown():
                    return None
                energy_required = 3
                base_damage = 140
                base_score = 50
                # Penalty when at risk (low prizes, opponent threatening)
                if self.my_prizes_left in {2, 3}:
                    op_active_threat = self.opponent.active[0] if self.opponent.active else None
                    if op_active_threat and len(op_active_threat.energies) >= 2:
                        base_score -= 200
                    else:
                        base_score -= 50
        else:
            return None

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        # Stonjourner is Basic - no evolution needed
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    op_data = card_table.get(op_pokemon.id)
                    if op_data is None:
                        continue
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    # Bonus for attacking ex Pokemon with Stonjourner (safe, high prize)
                    if my_pokemon.id == C.STONJOURNER and op_data.ex:
                        score += 200

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id == C.STONJOURNER:
            score += 100 if energy_count < 3 else 0
            score += 50 if energy_count < 1 else 0
            if active:
                score += 50  # prioritize active Stonjourner
        elif pokemon.id == C.ROLYCOLY:
            score -= 100  # low priority for energy

        if active and plan.attacker == 0 and plan.needs_energy:
            score += 200
        if self.my_prizes_left <= 2 and active:
            score += 50
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            # Don't retreat Stonjourner (it's our wall against ex decks)
            my_active_p = self.me.active[0] if self.me.active else None
            if my_active_p is not None and my_active_p.id == C.STONJOURNER:
                # Only retreat if on Boundless Power cooldown and bench has energy
                if self._is_boundless_power_on_cooldown():
                    for bp in self.me.bench:
                        if bp is not None and len(bp.energies) >= 1:
                            return 1500
                return -1
            if plan.attacker >= 1:
                return 2000
            if my_active_p is not None:
                my_data = card_table.get(my_active_p.id)
                my_maxhp = my_data.hp if my_data else my_active_p.hp
                if my_active_p.hp <= 80 and my_maxhp > 80:
                    for bp in self.me.bench:
                        if bp is not None and len(bp.energies) >= 1:
                            return 1500
            return -1
        if option.type == OptionType.ATTACK:
            # Find this option's attack index
            opt_idx = None
            for i, opt in enumerate(self.select.option):
                if opt is option:
                    opt_idx = i
                    break
            atk_idx = self._attack_option_map.get(opt_idx, -1) if opt_idx is not None else -1
            # Check if this matches the planned attack
            if plan.attack_index >= 0 and atk_idx == plan.attack_index:
                base = 1100
            else:
                base = 1000
            if plan.remain_hp <= 0:
                base += 200
            return base
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return self._score_setup_bench(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.TO_BENCH:
            return self._score_to_bench(card)
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}:
            return self._score_damage_counter(option, card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.STONJOURNER:
            score += 20
            # Higher priority when opponent has ex Pokemon
            op_active = self.opponent.active[0] if self.opponent.active else None
            if op_active is not None:
                op_data = card_table.get(op_active.id)
                if op_data is not None and (op_data.ex or op_data.megaEx):
                    score += 30
        elif card.id == C.ROLYCOLY:
            score += 4
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card is None:
            return 0
        if card.id == C.STONJOURNER:
            return 5
        if card.id == C.ROLYCOLY:
            return 3
        return 0

    def _score_setup_bench(self, card: Pokemon | Card) -> int:
        if card is None:
            return 0
        if card.id == C.STONJOURNER:
            return 120 - 25 * self.field_counts[C.STONJOURNER]
        if card.id == C.ROLYCOLY:
            return 80 - 20 * self.field_counts[C.ROLYCOLY]
        return 0

    def _score_to_bench(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None or data.cardType != CardType.POKEMON:
            return 0
        return self._score_setup_bench(card)

    def _score_discard(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        cid = card.id
        if cid == C.BASIC_FIGHTING_ENERGY:
            score = 45 if self.hand_counts[cid] >= 2 else 5
            if plan.needs_energy and not self.state.energyAttached:
                score -= 200
            if self.my_prizes_left <= 2:
                score -= 100
            return score
        if self.hand_counts[cid] >= 2:
            return 70
        if cid == C.GRAVITY_MOUNTAIN and self.stadium_id == C.GRAVITY_MOUNTAIN:
            return 50
        if cid in {C.CARMINE, C.LILLIE_DETERMINATION} and self.state.supporterPlayed:
            return 30
        if cid in {C.STONJOURNER, C.ROLYCOLY, C.BOSS_ORDERS}:
            return -40
        return 0

    def _score_damage_counter(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex == self.op_index:
            return 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0)
        return -target_score(card)

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.STONJOURNER:
            stonjourner_line = self.field_counts[C.STONJOURNER]
            score += -150 if stonjourner_line >= 2 else -3 if stonjourner_line >= 1 else 40
        elif card.id == C.ROLYCOLY:
            score += -100 if self.field_counts[C.ROLYCOLY] >= 2 else 30
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000
        if card.id == C.STONJOURNER and self.field_counts[C.STONJOURNER] >= 2:
            return -1
        if card.id == C.ROLYCOLY and self.field_counts[C.ROLYCOLY] >= 2:
            return -1
        if card.id == C.STONJOURNER and not self.has_ready_stonjourner:
            score += 500
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            if plan.remain_hp > 0 and plan.remain_hp <= 30:
                return 5500
            return 5000
        if card.id == C.BOSS_ORDERS:
            if plan.target >= 1:
                if plan.remain_hp <= 0:
                    return 3600
                return 3200
            # Snipe ex Pokemon on bench
            for op_p in self.opponent.bench:
                if op_p is None:
                    continue
                op_data = card_table.get(op_p.id)
                if op_data is not None and (op_data.ex or op_data.megaEx):
                    my_atk = self.me.active[0] if self.me.active else None
                    if my_atk is not None:
                        atk = self._base_attack(my_atk, 0)
                        if atk and op_p.hp <= atk[1]:
                            return 3000
            return -1
        if card.id == C.CARMINE:
            if self._low_deck():
                return -1
            if self.me.deckCount <= 12:
                return 2800
            return 3000
        if card.id == C.LILLIE_DETERMINATION:
            if self._low_deck():
                return -1
            if self.me.deckCount <= 15 and self.me.deckCount > LOW_DECK_COUNT:
                return 3300
            return 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and card_table.get(pokemon.id) is not None and card_table[pokemon.id].stage2
            for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        # Stonjourner and Rolycoly are Basic Pokemon - no evolution in this deck
        # Keep the function for safety (engine might offer evolve for other cards)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        return 9000 + len(pokemon.energies) * 2

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.LUMIOSE_CITY:
            return 1
        return 30000

    def remember_chosen_side_effects(self, selection: list[int]) -> None:
        global ability_used, boundless_power_turn
        if self.context != SelectContext.MAIN:
            return
        for idx in selection:
            if not (0 <= idx < len(self.select.option)):
                continue
            option = self.select.option[idx]
            if option.type == OptionType.ATTACK:
                # Check if this is Boundless Power (attack_index 1)
                atk_idx = self._attack_option_map.get(idx, -1)
                if atk_idx == 1:  # Boundless Power
                    boundless_power_turn = self.state.turn
                    _DIAG["boundless_power_used"] += 1


def agent(obs_dict: dict) -> list[int]:
    global pre_turn, ability_used, plan, boundless_power_turn

    # Deck-selection phase: the engine asks for the deck with select == None.
    # Return the 60-card id list (different return semantics from a normal turn).
    try:
        select_is_none = isinstance(obs_dict, dict) and obs_dict.get("select") is None
    except Exception:
        select_is_none = False
    if select_is_none:
        _DIAG["deck_returns"] += 1
        return my_deck

    _DIAG["decisions"] += 1
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _DIAG["deck_returns"] += 1
            _DIAG["decisions"] -= 1
            return my_deck

        if obs.current is not None and pre_turn != obs.current.turn:
            pre_turn = obs.current.turn
            ability_used = False
            plan = AttackPlan()

        try:
            policy = StonjournerPolicy(obs)
            _diag_observe(obs)
            ranked, scores = policy.rank()
            planned = search_plan(obs_dict, obs, ranked, scores) if USE_SEARCH else None
            ranked_use = ranked if planned is None else planned
            selection = normalize_selection(ranked_use, scores, obs.select)
            policy.remember_chosen_side_effects(selection)
            _diag_observe_choice(obs, selection)
            _DIAG["policy_ok"] += 1
            return selection
        except Exception as exc:
            _diag_record_error(exc)
            _DIAG["policy_fallback"] += 1
            return _legal_fallback(obs.select)
    except Exception as exc:
        _diag_record_error(exc)
        _DIAG["obs_fallback"] += 1
        return _legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})

