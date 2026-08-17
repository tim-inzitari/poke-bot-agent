# Slowking / ShumpeiNomura replay-distilled pilot guide

Status: research-only guide for the unique Slowking team in the 2026-08-04
top-ladder replay archive. It does not reopen, register, serve, submit, or grant
training authority to the terminally failed Slowking specialist.

Evidence boundary: the public index advertised 4,816 episodes, while 4,811 JSON
episodes were present in the downloaded daily archive. All 4,811 present files
were scanned; 159 contained Slowking SCR 58. All 159 Slowking seats used one exact
60-card multiset and the team name `ShumpeiNomura`. The archive is pinned as
`sha256:17cd9cd92f4ae3b293ee3fab3452657316362af134c6d4a7b5dbfda99c3d3d42`.
The complete derived evidence is in
`state/slowking_top_replay_distillation_2026-08-04.json`.

## The reconstructed list

Pokémon — 21

- 4 Slowpoke
- 4 Slowking
- 3 Mega Kangaskhan ex
- 2 Kyurem
- 2 Smoochum
- 2 Latias ex
- 1 Conkeldurr
- 1 Annihilape
- 1 Fezandipiti ex
- 1 Meowth ex

Trainer — 31

- 4 Academy at Night
- 4 Ciphermaniac’s Codebreaking
- 4 Lillie’s Determination
- 4 Night Stretcher
- 4 Poké Pad
- 4 Ultra Ball
- 3 Wondrous Patch
- 1 Colress’s Tenacity
- 1 Hilda
- 1 Prime Catcher
- 1 Switch

Energy — 8

- 4 Basic Psychic Energy
- 4 Telepath Psychic Energy

Exact multiset fingerprint:
`sha256:d3f092b737c0990149541576444e767d34a7367576d15a8f1b39f7b15460645d`.

This is not the older owner-shown list. The important changes are the third
Mega Kangaskhan ex, real gust and switch outs, single-copy Conkeldurr and
Annihilape, and the absence of Boomerang Energy, Counter Gain, and Secret Box.

## What the replay evidence says

The team went 86–73 (54.1%) in this one daily opponent mix. It was 45–25
(64.3%) when moving first and 41–48 (46.1%) when moving second. The intervals
are wide enough that these are descriptive, not a causal turn-order experiment,
but the direction agrees with the deck’s need to evolve and attach twice.

The strongest confirmed behavioral signals were:

- Of 67 recovered Seek Inspiration source cards, 44 were Kyurem (65.7%), 11
  Annihilape (16.4%), 10 Conkeldurr (14.9%), and 2 Slowking (3.0%).
- Academy at Night produced almost the same confirmed payload mix: 25 Kyurem,
  6 Annihilape, 5 Conkeldurr, and 3 Slowking. This directly supports the
  Academy → payload-on-top → Seek Inspiration sequence.
- Telepath Psychic Energy searched 52 Slowpoke, 22 Latias ex, and 9 Smoochum
  in 83 confirmed selections. Slowpoke was 62.7% of those targets.
- Poké Pad selected Slowking 27 times and Slowpoke 24 times in 64 confirmed
  selections. The evolution engine accounted for 79.7% of its targets.
- Of 333 confirmed manual attachments, 100 went to Mega Kangaskhan ex, 78 to
  Slowpoke, 54 to Smoochum, 46 to Latias ex, and 37 to Slowking. Only four
  went to Kyurem. Energy is principally building the active engine or the next
  attacker; Kyurem is normally a copied-attack payload, not an attacker.
- Among 108 confirmed attack actions, 67 came from Slowking, 40 from Smoochum,
  and one from Slowpoke. Delightful Kiss is a material early-game line, not a
  decorative attack.
- In 127 opening-Active choices that could be confirmed from subsequent state
  transitions, Mega Kangaskhan ex appeared 48 times, Smoochum 31, Slowpoke 21,
  Latias ex 18, Kyurem 6, and Meowth ex 3. These are availability-conditioned
  observations, not unconditional preference probabilities.

## Pilot plan

### 1. Prefer the first-turn evolution clock

When given the choice, the default is to move first: establish Slowpoke and
attach on turn one, then evolve, attach again, stack the payload, and attack on
turn two. The daily replays do not randomize turn order, so this remains a
strong practical rule rather than a proved causal estimate.

Going second is a real Smoochum mode. Delightful Kiss can accelerate the Basic
Psychic plan while a first-turn Supporter develops the board. Do not pretend
that going second follows the same clock as going first.

### 2. Choose the opening Active by job

Use Mega Kangaskhan ex when its early consistency work is worth exposing a
multi-Prize Pokémon and you already have, or can find, a pivot. The list now
has three Kangaskhan, two Latias ex, Switch, and Prime Catcher, so the opening
Kangaskhan line is much more intentional than in the older build.

Use Smoochum when Delightful Kiss is the turn-one plan. Use Slowpoke Active only
when the hand can protect the evolution/attachment clock or when no better
pivot exists. Avoid opening Kyurem: it is far more valuable as a Seek
Inspiration source, and the confirmed replays almost never invested manual
attachments into it.

### 3. Build the Slowpoke chain before luxury support

The default board needs two Slowpoke lines. Telepath Psychic Energy and Poké
Pad both overwhelmingly serviced Slowpoke/Slowking development. The first line
is the immediate attacker; the second is continuity after Trifrost discards
Energy or the Active is Knocked Out.

Bench support only with a named job:

- Latias ex: free-retreat/pivot bridge.
- Mega Kangaskhan ex: setup consistency or a deliberate direct attacker.
- Smoochum: Delightful Kiss acceleration.
- Fezandipiti ex: recovery draw after a Knock Out.
- Meowth ex: its exact current-state job, not generic bench fill.

Keep an open slot for the next Slowpoke, a recovery target, or a forced pivot.
Prime Catcher makes prize-map awareness on both benches more important.

### 4. Sequence every shuffle before the stack

The default order is:

1. inventory the deck and prizes through legal searches;
2. bench and evolve;
3. use Ultra Ball, Poké Pad, Telepath Psychic Energy, Hilda, Colress’s
   Tenacity, Lillie’s Determination, and other shuffle effects;
4. use draw/consistency Abilities;
5. solve attachment, Wondrous Patch, retreat, Switch, and Prime Catcher;
6. use Academy at Night as the final top-card placement;
7. move Slowking Active;
8. use Seek Inspiration.

A draw consumes the stacked card. A shuffle destroys the stack. Hilda and
Colress are especially easy to misuse because they fetch exactly the pieces
the turn wants while also shuffling: use them before Academy.

Ciphermaniac has two modes. The simple mode puts the attack payload on top and
the next-turn card beneath it. The stronger bridge mode uses Mega Kangaskhan to
draw the two Ciphermaniac cards, then Academy restacks the payload before the
pivot to Slowking. Do not Run Errand after stacking unless Academy remains
available to rebuild the top card.

### 5. Select the copied attack from the prize map

The replay policy was Trifrost-first but not Trifrost-only.

Choose Kyurem when 110 damage to three targets immediately takes multiple
Prizes, removes three setup Pokémon, or creates a clearly superior two-turn
map. Confirm all three targets before committing. Because this list has no
Boomerang Energy, Trifrost’s Energy discard is a real resource loss; Wondrous
Patch can recover Basic Psychic to the Bench, but it cannot recover Telepath
Psychic as Basic Energy.

Choose Conkeldurr when 250 damage cleanly removes the Active and spread would
not create more value. There is only one copy, so know whether it is in hand,
deck, discard, or Prizes before planning around it.

Choose Annihilape when damage is the wrong axis: an oversized Active, damage
prevention, or a decisive simultaneous Knock Out. Check both Prize counts and
replacement attackers before using Destined Fight. This is also a singleton.

The two observed Slowking-on-Slowking copies are rare fallback evidence, not a
core line. Do not infer recursive Seek Inspiration behavior beyond what the
current engine and ruling actually allow.

### 6. Treat movement and gust as combo pieces

Switch and Prime Catcher materially change the older pilot. They let a setup
Kangaskhan leave the Active Spot, place Slowking Active without spending its
Energy, and choose the opposing Active that best matches Trifrost, Gutsy Swing,
or Destined Fight. Resolve gust and pivot before the final Academy stack.

Latias ex is still the reusable mobility engine, but the list is no longer
completely dependent on finding it. Do not expose Latias merely because it is
searchable; Switch can sometimes preserve the Bench and Prize map.

### 7. Budget eight Energy across repeated attacks

There are only eight Energy and no Counter Gain. A normal Slowking needs two
attachments’ worth of cost. The next attacker should be developing before the
current attack whenever possible.

The confirmed attachment distribution shows three real branches:

- Kangaskhan/Smoochum/Latias for the opening engine;
- Slowpoke/Slowking for the current and next Seek Inspiration;
- very little direct Kyurem investment.

Wondrous Patch is the main post-Trifrost rebuild. Preserve Basic Psychic in the
discard and a Bench target for it. Night Stretcher is unusually important at
four copies because it restores the single Conkeldurr or Annihilape, a broken
Slowpoke line, or Basic Energy according to the exact current need.

## Fast decision checklist

Before every attack, answer:

1. What exact Prize map do Kyurem, Conkeldurr, and Annihilape produce now?
2. Is the selected singleton payload available and recoverable?
3. Have all draws and shuffles already happened?
4. Is Academy still available for the final stack?
5. Can Slowking reach the Active Spot without discarding needed Energy?
6. Which Bench Pokémon becomes the next attacker, and how does it get two
   Energy?
7. What is the opponent’s shortest route through the multi-Prize Bench?
8. If using Trifrost, which three targets and which next-turn knockouts justify
   discarding all Energy?
9. If using Destined Fight, who wins after both Active Pokémon are Knocked Out?

## What not to copy blindly from the bot

Replay frequencies do not expose unchosen counterfactuals. They are conditional
on the bot’s hand, its exact simulator observation, and the day’s opponent mix.
The 54.1% daily win rate is not a formal superiority result, and team names are
not normalized matchup labels. Use the highly repeated causal patterns—engine
development, final stacking, copied-attack distribution, Energy routing—as a
teacher. Do not turn raw action frequency into unconditional rules.

## Sources paired into this guide

- Kaggle daily top-ladder archive,
  `pokemon-tcg-ai-battle-episodes-2026-08-04`.
- Derived replay receipt,
  `state/slowking_top_replay_distillation_2026-08-04.json`.
- Existing card-text and strategy synthesis,
  `docs/deck_guides/slowking-expert-brief.txt`.
- Existing causal/abstaining heuristic audit,
  `state/slowking_heuristic_research_v1.txt`.
