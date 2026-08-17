# Alakazam new-list direct-policy pilot guide

## Identity and purpose

This guide covers the owner's exact 60-card Powerful Hand Alakazam list bound
to canonical multiset digest
`sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182`.
Its core is four Abra, four Kadabra, three Alakazam, three Dunsparce, two
Dudunsparce, one Fezandipiti ex, and the Trainer and Energy engine listed below.
The model guide is training-only. For a human pilot, the same ideas are a
decision framework: build a replacement attacker, grow the hand only as far as
the prize map requires, and keep enough recovery and deck depth to finish.

### Exact deck

Pokémon (17): 4 Abra, 4 Kadabra, 3 Alakazam, 3 Dunsparce, 2 Dudunsparce,
1 Fezandipiti ex.

Trainers (36): 4 Battle Cage, 4 Buddy-Buddy Poffin, 4 Dawn, 4 Enhanced Hammer,
4 Hilda, 4 Poké Pad, 3 Rare Candy, 2 Night Stretcher, 3 Boss's Orders,
2 Xerosic's Machinations, 1 Lana's Aid, 1 Sacred Ash.

Energy (7): 4 Telepath Psychic Energy, 2 Basic Psychic Energy,
1 Enriching Energy.

## The central damage equation

Powerful Hand places two damage counters for every card in hand. Convert the
opposing Active Pokémon's remaining HP into a hand-size target before spending
anything: required cards = ceiling(remaining HP / 20). Weakness, Resistance,
and ordinary damage modifiers do not change placed damage counters. Visible
effects that prevent effects of attacks, including Mist Energy and Rock
Fighting Energy where applicable, can stop the counters and must be removed or
played around. Recalculate after every draw, discard, gust, switch, heal, or
other public state change.

Once the hand reaches the exact knockout threshold, stop treating more cards
as automatically better. Every additional search or draw can expose the deck
to decking out, consume the recovery card needed for the next attacker, or
spend a Supporter that would be stronger after the prize exchange. The goal is
not the largest hand. It is the smallest safe hand that wins the current prize
exchange while preserving the next one.

## Opening setup

When choosing an Active, prefer Abra when it can safely become the first
Alakazam. Dunsparce is the fallback when Abra would be trapped or immediately
lost without advancing the board. During bench setup, establish a second Abra
before Fezandipiti ex. Develop one Dunsparce line early enough to access
Dudunsparce draw and recycle, but do not fill the Bench with redundant draw
pieces. Keep a Bench slot open when a later replacement attacker or
Fezandipiti ex may be required.

Your minimum stable board is:

1. an Abra line that can become the current attacker;
2. a second Abra or evolved line that can become the replacement attacker;
3. one Dunsparce line when its draw and recycle materially improves the next
   two turns; and
4. at least one flexible Bench slot unless the winning line is already exact.

Do not Rare Candy an Abra merely because the play is legal. Confirm that the
Alakazam can attack, that its draw effect will not overdraw the deck, and that
using Rare Candy does not leave the replacement line impossible. Evolving
through Kadabra is often correct when it preserves Rare Candy, adds controlled
draw, or keeps multiple evolution paths live.

## Going first

The final package chooses first whenever the rules expose the choice. Use the
extra development turn to establish Abra plus a replacement line, then add the
Dunsparce engine. Attach Energy to the line most likely to survive and attack
on the following turn. Avoid committing every resource to the Active Abra if
the opponent can take it before it evolves.

The ideal first-turn sequence is setup search, board development, then only the
draw needed to confirm next turn's evolution and attachment. Preserve Dawn,
Hilda, Poké Pad, and recovery cards until their exact search or draw value is
known. End with a clear next-turn map: evolution card, Energy, target hand size,
and a replacement-attacker path.

## Going second

Going second does not change the deck's identity, but it compresses setup.
Prioritize a legal attacker line and an Energy attachment over speculative hand
growth. If an early attack is impossible, spend the turn creating two live
evolution routes rather than maximizing the current hand. Accept a smaller
first attack when it preserves the board and creates a two-hit prize line the
opponent cannot efficiently answer.

## Search and draw sequencing

Sequence deterministic search before broad draw when the search target is
known. Count the remaining copies of Abra, Kadabra, Alakazam, Rare Candy,
Energy, and recovery before selecting a card. Then draw only after removing the
exact wanted card from the deck. Reverse that order only when the draw can
materially change which search target is correct.

Use Dudunsparce when the extra cards solve a defined problem: reaching knockout
hand size, finding the evolution, finding Energy, or finding the recovery line.
Its recycle value matters. Do not cash it in when the current hand already
attacks for the required knockout and the additional draw only raises deck-out
risk.

Telepath Psychic Energy is a setup attachment: it searches Basic Psychic
Pokémon directly to the Bench rather than drawing them into the hand. Enriching
Energy is the attachment that draws four cards. Account for Enriching Energy,
Psychic Draw, Flip the Script, and Run Away Draw before committing to them.
Near the end of the game, leave enough cards for the mandatory start-of-turn
draw and any draw effect the planned line cannot avoid. A powerful attachment
that empties the deck is not safe unless the game ends before the next draw.

Do not play Enriching Energy when it will be the only Energy in play unless no
other draw exists and there is no reasonable path to a different Energy that
turn. It is usually safe on turn one, going first or second, when its draw is
needed. Remember that attaching it means not attaching Psychic Energy that
turn. Without Psychic Energy on Abra, Kadabra, or Alakazam, attacking that turn
may be impossible. Prioritize taking prizes and winning. Attach Enriching
Energy to Dudunsparce or Dunsparce when possible so Runaway Draw can recycle
the resource.

## Attack and prize planning

Before each attack, write the prize map mentally:

- What does Powerful Hand knock out now?
- How many prizes does that target yield?
- Would Boss's Orders on a Benched Pokémon reduce the turns needed to win?
- What attacker can the opponent return with?
- Which Alakazam line attacks after the return knockout?
- How many hand cards will that next knockout require?

Boss's Orders is for a prize-changing or tempo-winning target, not merely a
legal Bench target. Gust when it creates a knockout, strands a costly retreat,
removes a draw or Energy engine that changes the next turn, or closes the game.
Otherwise preserve Boss for the point where it changes the prize map.

Kadabra can be a finisher when its attack reaches a small remaining-HP total
without exposing Alakazam or consuming the hand needed for the next prize.
Treat that as an exact damage line, not an automatic promotion of Kadabra to
the Active Spot.

## Resource and recovery rules

Night Stretcher, Sacred Ash, and Lana's Aid are continuity cards. Night
Stretcher returns one Pokémon or Basic Energy; Lana's Aid returns up to three
non-Rule-Box Pokémon and Basic Energy in combination; Sacred Ash shuffles up to
five Pokémon into the deck. Night Stretcher and Lana's Aid cannot recover the
Special Energy in this list, and Lana's Aid cannot recover Fezandipiti ex.
Before using one, identify whether the missing resource is a Basic, an
Evolution Pokémon, Basic Energy, or multiple Pokémon that must return to the
deck. Choose the narrowest recovery effect that restores the actual
next-attacker path.

Enhanced Hammer is conditional disruption. Use it when removing the visible
Special Energy meaningfully delays the opponent, breaks an attack, removes
protection from Powerful Hand, or changes the prize race. Do not spend it
because a Special Energy is merely present. Xerosic's Machinations is a timing
tool whose value depends on the opponent's public hand size and board needs.

## Bench and prize discipline

Keep two Alakazam lines possible until the remaining prize map proves one is
enough. Avoid benching Fezandipiti ex unless a publicly observed knockout
during the opponent's previous turn makes Flip the Script available and its
draw is required to repair the hand or complete a decisive attack. Every
two-prize support Pokémon changes the opponent's route to six prizes.

Track prized pieces through absence: if a four-copy card is missing from the
deck, hand, field, and discard after enough cards are known, treat it as
possibly prized. Do not assume a hidden prize identity. Build the line that
works under the public worst case until a prize reveal supplies exact
information.

## Disruption recovery

After hand disruption, rebuild in this order:

1. confirm a legal attacker or evolution line;
2. confirm an Energy attachment or retreat requirement;
3. restore the hand to the current knockout threshold;
4. restore the replacement attacker; and
5. add optional disruption or utility.

Do not chase the former hand size. Recompute the required hand from the new
Active target and prize state. If a one-turn knockout is no longer safe, choose
a two-hit line that preserves the board.

## Common errors

- Drawing past the exact Powerful Hand threshold and losing to deck-out.
- Spending Rare Candy without a replacement evolution route.
- Filling the Bench before the opponent's prize plan is known.
- Using Boss, Enhanced Hammer, or Xerosic only because it is legal.
- Attaching Enriching Energy without counting forced draws and next-turn deck
  size.
- Recovering the largest number of cards instead of the exact missing combo.
- Treating hidden prizes or the opponent's future hand as known information.
- Building one oversized hand while allowing the only attacker line to fail.

## Turn checklist

At every decision, answer:

1. What exact hand size knocks out the defending Pokémon?
2. Which cards can be spent without falling below it?
3. Is the next Alakazam line already live?
4. How many unavoidable draws remain before the next attack?
5. Does this Bench card improve the prize race enough to expose it?
6. Does this disruption change an attack, retreat, or prize outcome now?
7. What is the safest line if one unknown key card is prized?
8. Can the game end before the deck must draw again?

## Matchup-specific rules

In the Alakazam mirror, preserve Xerosic for a turn when reducing the
opponent's hand meaningfully cuts Powerful Hand damage. A strong timing point
is the turn an opposing Alakazam is knocked out, forcing the opponent to spend
cards rebuilding both attacker and hand.

When the opponent plays Xerosic, discard cards that are not useful in that
matchup. Enhanced Hammer is expendable against decks that have not shown Mist
Energy or Rock Fighting Energy. Xerosic is expendable when the opposing board
does not show an Alakazam plan. Be cautious about ending above seven cards in
the mirror. Draw what is necessary to repair a board with two or fewer live
Alakazam lines or to knock out Fezandipiti ex, but aim to preserve the roughly
seven-card threshold needed to place fourteen counters. Leaving Dudunsparce in
play is a useful recovery route after opposing Xerosic.

Against Cynthia's Garchomp ex, preserve Enhanced Hammer for Rock Fighting
Energy. Its attack-effect protection stops Powerful Hand's counters; remove it
before attacking the attached Pokémon.

Against Crustle, prioritize Mist Energy with Enhanced Hammer unless a different
Hammer target immediately wins. Against any deck that has shown Mist Energy or
Rock Fighting Energy, prioritize those cards when attached to the opposing
Active Pokémon. If they remain attached, gusting around the protected target
may be the only winning line.

Battle Cage is a defensive tool in matchups where it blocks damage counters
placed on the Bench by effects of opposing attacks or Abilities. It does not
stop ordinary attack damage to the Bench. Prioritize it when opposing Froslass
or Munkidori is already in play and a relevant Benched Pokémon can be
protected. Do not play it early merely because it is available. Replacing an
opponent's Stadium is often more useful. When the opponent does not target the
Bench, use Battle Cage primarily as a timed Stadium denial card.

## Source basis

This is the owner's supplied guide, adapted only to the exact new list and its
public card mechanics. It retains the supplied evidence set:

- TCGplayer, “Alakazam Deck Guide — Pokémon TCG”:
  https://www.tcgplayer.com/content/article/Alakazam-Deck-Guide-Pok%C3%A9mon-TCG/7eb46b82-9dc5-40d8-adf9-28cca05f070f/
- PTCGO News, “PTCGL Alakazam Deck Guide”:
  https://ptcgonews.com/tips/ptcgl-alakazam-deck-guide/
- Kaggle Pokémon TCG AI Battle discussion 717697:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717697

## Training appendix

The model guide is curriculum metadata for the r241 direct policy, not a
runtime policy. On guide-qualified rows it weights observed causal strategic
targets at the receipt-bound multiplier. It never supplies an action logit,
replaces an observed outcome, reads hidden or future information, invokes MCTS,
or invokes RTP. Ambiguous or incomplete legal stages are entirely masked.
