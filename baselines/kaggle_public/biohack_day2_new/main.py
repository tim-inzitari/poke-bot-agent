import os
from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectType, SelectContext,
    OptionType, Card, Pokemon, State, all_card_data, all_attack, to_observation_class,
)

# =============================================================================
# Crustle wall agent  (PTCG AI Battle Challenge / cabt engine)
#
# Design philosophy is unchanged from the day-1 #1 bot: the deck does the work,
# the pilot stays simple and *robust*. Every decision is a score; we sort the
# legal options and return the best. The score is mostly a PRIORITY ORDER:
#
#     ATTACH (1000) > EVOLVE (800) > PLAY (600) > ABILITY (400)
#                   > ATTACK (100..) > END (0) > RETREAT (<0)
#
# i.e. do all your setup first, attack last (attacking ends the turn).
#
# What this version adds on top of the proven baseline — all of it defensive,
# none of it able to crash the match:
#   1. Hard crash-proofing: the whole body runs inside a guard that, on ANY
#      unexpected error, returns a guaranteed-legal fallback action. A bot that
#      never errors out beats a clever bot that occasionally forfeits.
#   2. Card / attack metadata (all_card_data, all_attack) loaded once so we can
#      reason about damage, prize value, evolution stage, etc.
#   3. Smarter ATTACK choice: among legal attacks, prefer one that KOs the
#      opponent's Active, otherwise the higher-damage one (still ranked below
#      setup actions so we always set up first).
#   4. Smarter ENERGY attachment: concentrate energy on the Active wall/attacker
#      (and on a benched Crustle next), which also brings Jumbo Ice Cream online.
#   5. Smarter forced sub-selections: KO/prize-aware opponent targeting, heal the
#      most-damaged mon, and when forced to discard, throw away junk (spare basic
#      energy) and protect the key pieces (Crustle / Dwebble / Hero's Cape).
#
# New in this revision — driven by the day-1 mirror replay (episode 80358072),
# which was lost to an empty bench ("no Active Pokemon", reason 3) with 6 prizes
# still up:
#   6. Bench insurance: if a play benches a Basic Pokemon and our bench is
#      empty/thin, do it early; and when searching/fetching, grab a benchable
#      Basic first when the bench is bare. This directly defends the loss
#      condition that decided that game.
#   7. Targeted retreat: if we're stuck with a Dwebble in the Active spot while a
#      Crustle sits benched, promote the wall (the engine only ever offers RETREAT
#      when it is legal/payable, so this is safe). Otherwise we never retreat.
#   8. Turn order: a slow heal/wall deck prefers to go SECOND (it can use a
#      Supporter and respond from turn 1), so on the IS_FIRST choice we pick the
#      "go second" option. See PREFER_GO_FIRST below — flip it if your runtime's
#      YES/NO semantics turn out inverted.
# =============================================================================

# ---- card IDs in the deck (see deck.csv) ------------------------------------
DWEBBLE = 344
CRUSTLE = 345
BUDDY_POFFIN = 1086
JUMBO_ICE_CREAM = 1147
HEROS_CAPE = 1159
COOK = 1212
CHEREN = 1224
BATTLE_CAGE = 1264
BASIC_GRASS = 1
GROW_GRASS = 18
MIST = 11
SPIKY = 14

KEY_PIECES = {CRUSTLE, DWEBBLE, HEROS_CAPE}            # never discard if avoidable
USEFUL_PIECES = {JUMBO_ICE_CREAM, COOK, CHEREN, BUDDY_POFFIN, BATTLE_CAGE}

# Turn-order preference. A slow heal/wall deck generally prefers to go SECOND: it
# can play a Supporter (Cheren/Cook) and respond on its first turn, whereas the
# player on the play cannot. Set True only if leaderboard testing shows the
# IS_FIRST YES/NO semantics are inverted in your runtime.
PREFER_GO_FIRST = False

# Contexts that mean "we are putting / fetching a Pokemon into play" -> prefer wall pieces.
PLACEMENT_CTX = {
    SelectContext.EVOLVE, SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO,
    SelectContext.TO_BENCH, SelectContext.TO_FIELD, SelectContext.TO_ACTIVE,
    SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
}
# Contexts that mean "we are giving a card up" -> dump junk, keep key pieces.
GIVE_UP_CTX = {
    SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD, SelectContext.TO_PRIZE,
}
# Contexts where we want to heal/repair the most-damaged Pokemon.
HEAL_CTX = {SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER}


# ---- one-time metadata tables (safe if the API is unavailable) --------------
try:
    _CARD = {c.cardId: c for c in all_card_data()}
except Exception:
    _CARD = {}
try:
    _ATK = {a.attackId: a for a in all_attack()}
except Exception:
    _ATK = {}


def card_meta(card_id):
    try:
        return _CARD.get(card_id)
    except Exception:
        return None


def attack_meta(attack_id):
    try:
        return _ATK.get(attack_id)
    except Exception:
        return None


def read_deck_csv():
    """Return the 60 card IDs for the deck (works in notebook or match runtime)."""
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + path
    with open(path, "r") as f:
        lines = [ln for ln in f.read().splitlines()]
    return [int(lines[i]) for i in range(60)]


def get_card(obs, area, index, player_index):
    """Safely fetch a Card/Pokemon from a zone; never raises."""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def _opp_active(obs, me):
    try:
        op = obs.current.players[1 - me]
        if op.active and op.active[0] is not None:
            return op.active[0]
    except Exception:
        pass
    return None


def opponent_value(p):
    """How tempting is this opponent Pokemon as a target (prize + investment)."""
    v = 0
    cd = card_meta(getattr(p, "id", None))
    if cd is not None:
        if getattr(cd, "megaEx", False):
            v += 300
        elif getattr(cd, "ex", False):
            v += 200
        if getattr(cd, "stage2", False):
            v += 120
        elif getattr(cd, "stage1", False):
            v += 60
    try:
        v += len(p.energies) * 40
    except Exception:
        pass
    try:
        v += len(p.tools) * 30
    except Exception:
        pass
    v += getattr(p, "hp", 0) // 10
    return v


def _my_active(obs, me):
    """Our Active Pokemon, or None."""
    try:
        a = obs.current.players[me].active
        if a and a[0] is not None:
            return a[0]
    except Exception:
        pass
    return None


def _my_bench_count(obs, me):
    """How many Pokemon are on our bench. On bad data, return a high number so we
    do NOT spuriously trigger the (emergency) bench-insurance behaviour."""
    try:
        return len(obs.current.players[me].bench)
    except Exception:
        return 99


def _bench_has_id(obs, me, card_id):
    """True if a Pokemon with this id sits on our bench."""
    try:
        for b in obs.current.players[me].bench:
            if b is not None and getattr(b, "id", None) == card_id:
                return True
    except Exception:
        pass
    return False


def _is_basic_pokemon(card_id):
    """True if this card is a Basic Pokemon (i.e. it can be played straight to the
    bench). Uses card metadata, with a fallback to the known Basic in this deck."""
    cd = card_meta(card_id)
    if cd is not None:
        try:
            return bool(getattr(cd, "basic", False)) and getattr(cd, "cardType", None) == CardType.POKEMON
        except Exception:
            pass
    return card_id == DWEBBLE


def _score_main(obs, o, me):
    """Score a MAIN-menu option. Keeps the proven priority + special cases."""
    t = o.type

    if t == OptionType.ATTACH:
        card = get_card(obs, o.area, o.index, me)
        # Hero's Cape: only ever onto the Active Pokemon (proven special case).
        if card is not None and getattr(card, "id", None) == HEROS_CAPE:
            return 2100 if o.inPlayArea == AreaType.ACTIVE else 0
        # Otherwise attach energy; prefer the Active wall, then a benched Crustle.
        if o.inPlayArea == AreaType.ACTIVE:
            return 1060
        if o.inPlayArea == AreaType.BENCH:
            tgt = get_card(obs, o.inPlayArea, o.inPlayIndex, me)
            if tgt is not None and getattr(tgt, "id", None) == CRUSTLE:
                return 1030
            return 1010
        return 1000

    if t == OptionType.EVOLVE:
        return 800

    if t == OptionType.PLAY:
        card = get_card(obs, AreaType.HAND, o.index, me)
        cid = getattr(card, "id", None) if card is not None else None
        active = None
        try:
            a = obs.current.players[me].active
            if a and a[0] is not None:
                active = a[0]
        except Exception:
            active = None
        # Jumbo Ice Cream: heal only when damaged AND has 3+ energy (proven).
        if cid == JUMBO_ICE_CREAM:
            if active is not None and active.hp < active.maxHp and len(active.energies) >= 3:
                return 2000
            return -2   # below END(0): never waste a heal that would do nothing
        # Cook: heal only when damaged (proven).
        if cid == COOK:
            if active is not None and active.hp < active.maxHp:
                return 1500
            return -2   # below END(0): don't burn our Supporter for a 0 heal
        if cid == CHEREN:      # draw 3 — keeps the engine flowing
            return 1400
        if cid == BATTLE_CAGE:  # stadium — protects the bench
            return 1300
        # Bench insurance: a benched backup is exactly what prevents a "no Active
        # Pokemon" loss (the way the day-1 mirror was lost). If this play drops a
        # Basic onto an empty/thin bench, prioritise it.
        if _is_basic_pokemon(cid):
            bench_n = _my_bench_count(obs, me)
            if bench_n <= 0:
                return 1700     # empty bench -> securing a backup is urgent
            if bench_n == 1:
                return 700      # thin bench -> mild bias to keep stocking it
            return 600
        return 600

    if t == OptionType.ABILITY:
        return 400

    if t == OptionType.ATTACK:
        # Attack last, but pick the BEST attack. Stays in the 100..~300 band so it
        # never outranks setup actions (ABILITY=400) — we always set up first.
        score = 100.0
        a = attack_meta(o.attackId)
        dmg = getattr(a, "damage", 0) if a is not None else 0
        try:
            dmg = int(dmg)
        except Exception:
            dmg = 0
        score += min(max(dmg, 0), 250) * 0.2          # 0..50, prefers higher damage
        opp = _opp_active(obs, me)
        if opp is not None and dmg > 0 and dmg >= getattr(opp, "hp", 10 ** 9):
            score += 150                               # this attack KOs the Active
        return score

    if t == OptionType.RETREAT:
        # Default: the wall stays put. BUT if we are stuck with a Dwebble Active
        # while a Crustle sits on the bench, promote the wall. The engine only
        # offers RETREAT when it is currently legal (retreat cost is payable), so
        # we don't need to model energy here.
        active = _my_active(obs, me)
        if (active is not None and getattr(active, "id", None) == DWEBBLE
                and _bench_has_id(obs, me, CRUSTLE)):
            return 120     # above a chip ATTACK -> swap the wall in first
        return -1          # otherwise the wall wants to stay Active

    if t == OptionType.END:
        return 0           # explicit fallback once nothing better remains

    # DISCARD or anything else on the main menu: low but valid.
    return 0


def _score_sub(obs, o, me, context):
    """Score a forced sub-selection. Base is high so we always make a valid choice."""
    t = o.type
    score = 2000.0

    if t == OptionType.CARD:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        if card is not None:
            cid = getattr(card, "id", None)

            if context in PLACEMENT_CTX:
                score += 500
                if cid == CRUSTLE:
                    score += 120        # get the wall down / evolved
                elif cid == DWEBBLE:
                    score += 80
                # Empty bench -> a benchable Basic is the priority fetch (insurance).
                if _is_basic_pokemon(cid) and _my_bench_count(obs, me) <= 0:
                    score += 400
            elif context == SelectContext.TO_HAND and cid in (CRUSTLE, DWEBBLE):
                score += 100            # searching to hand -> grab wall pieces
                if cid == DWEBBLE and _my_bench_count(obs, me) <= 0:
                    score += 200        # grab a benchable Basic when the bench is bare

            if isinstance(card, Pokemon):
                if o.playerIndex != me:
                    # Opponent target: prefer their Active and high-value mons.
                    score += 500 if o.area == AreaType.ACTIVE else 100
                    score += opponent_value(card)
                else:
                    if context in HEAL_CTX:
                        # heal where it matters most
                        score += max(0, getattr(card, "maxHp", 0) - getattr(card, "hp", 0))
                    else:
                        score += getattr(card, "hp", 0)
                        if cid == CRUSTLE:
                            score += 60   # protect / value the wall
            else:
                # A plain card (e.g. choosing what to discard / send to deck).
                if context in GIVE_UP_CTX:
                    if cid in KEY_PIECES:
                        score -= 300      # keep these — discarded only if forced
                    elif cid in USEFUL_PIECES:
                        score -= 80
                    elif cid == BASIC_GRASS:
                        score += 60       # spare basic energy is the most expendable

    elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
        # Choosing an attached energy (e.g. to discard / move). If we can tell it's
        # spare basic Grass, prefer to spend that over special energy.
        cid = getattr(o, "cardId", None)
        if context in (SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_ENERGY,
                       SelectContext.TO_HAND_ENERGY, SelectContext.TO_DECK_ENERGY,
                       SelectContext.DETACH_FROM):
            if cid == BASIC_GRASS:
                score += 40
            elif cid in (GROW_GRASS, MIST, SPIKY):
                score -= 40

    elif t == OptionType.YES:
        # Default to "yes" (activate effects, accept beneficial prompts). EXCEPT the
        # turn-order choice: a slow heal/wall deck prefers to go SECOND, so we
        # down-weight the "go first" option there.
        if context == SelectContext.IS_FIRST and not PREFER_GO_FIRST:
            score += 0
        else:
            score += 100

    elif t == OptionType.NO:
        # Mirror of the above: on the turn-order choice, prefer "no" (go second).
        if context == SelectContext.IS_FIRST and not PREFER_GO_FIRST:
            score += 150
        else:
            score += 0

    elif t == OptionType.NUMBER:
        # Generally take the largest offered number (draw more / place more counters).
        score += getattr(o, "number", 0) or 0

    elif t == OptionType.SPECIAL_CONDITION:
        score = 2000          # any valid choice keeps the game moving

    return score


def _legal_fallback(select):
    """A guaranteed-valid selection: the fewest required options, lowest indices."""
    try:
        n = len(select.option)
        lo = select.minCount if select.minCount is not None else 1
        hi = select.maxCount if select.maxCount is not None else 1
        k = max(0, min(lo, hi, n))
        return list(range(n))[:k]
    except Exception:
        return [0]


def agent(obs_dict: dict) -> list[int]:
    """Entry point. Returns option indices; length in [minCount, maxCount], unique.

    The entire body is guarded: any unexpected error falls back to a legal move,
    so the agent can never forfeit a game by raising.
    """
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        # Could only really happen at deck selection if conversion misbehaves.
        try:
            return read_deck_csv()
        except Exception:
            return []

    # Initial deck-selection phase.
    if obs.select is None:
        try:
            return read_deck_csv()
        except Exception:
            return []

    select = obs.select
    try:
        me = obs.current.yourIndex
    except Exception:
        me = 0
    context = getattr(select, "context", None)
    options = select.option

    try:
        is_main = (context == SelectContext.MAIN) or (
            getattr(select, "type", None) == SelectType.MAIN
        )
        scores = []
        for o in options:
            try:
                s = _score_main(obs, o, me) if is_main else _score_sub(obs, o, me, context)
            except Exception:
                s = 0  # a single bad option never sinks the whole turn
            scores.append(s)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        out = []
        limit = min(len(order), select.maxCount)
        for i in range(limit):
            idx = order[i]
            # keep negative-scored options only while we still owe minCount picks
            if scores[idx] >= 0 or len(out) < select.minCount:
                out.append(idx)

        # Safety net: respect minCount even if every option scored negative.
        if len(out) < select.minCount:
            for idx in order:
                if idx not in out:
                    out.append(idx)
                    if len(out) >= select.minCount:
                        break
        return out[: select.maxCount]
    except Exception:
        return _legal_fallback(select)
