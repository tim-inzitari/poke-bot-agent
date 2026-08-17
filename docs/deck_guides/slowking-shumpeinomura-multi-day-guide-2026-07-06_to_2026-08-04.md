# Slowking / ShumpeiNomura multi-day replay-distilled guide

Status: research-only guide for the current Slowking list used by the
`ShumpeiNomura` top-ladder bot. It does not reopen, register, serve, submit, or
grant training authority to the terminally failed Slowking specialist.

Evidence boundary: eight relevant daily archives were scanned across July 6–7
and July 30–August 4. Their manifests advertised 37,904 episodes and the ZIPs
contained 37,894 JSON members; all present files were scanned. This confirms
768 Slowking seats across two named teams and three exact lists:

- `vibechu`: 311 games with the older `56ac56…` list on July 6–7;
- `ShumpeiNomura`: 17 games with an intermediate `d44f70…` list on July 31–August 1; and
- `ShumpeiNomura`: 440 games with the final `d3f092…` list on August 2–4.

The aggregate evidence is in
`state/slowking_multi_day_replay_distillation_2026-07-06_to_2026-08-04.json`.
This guide uses the 440 final-list games for action frequencies, while the 328
earlier games are a separate transfer and list-evolution stratum.

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

This is not the older `vibechu` list. The important changes are the third
Mega Kangaskhan ex, real gust and switch outs, single-copy Conkeldurr and
Annihilape, and the absence of Boomerang Energy, Counter Gain, and Secret Box.
The 17-game intermediate build was one card away: one Spectrier in place of the
second Smoochum.

## What the replay evidence says

The final list went 258–182 (58.6%) over three days. It was 135–81 (62.5%)
when moving first and 123–101 (54.9%) when moving second. The 95% Wilson
intervals are 54.0–63.1% overall, 55.9–68.7% first, and 48.4–61.3% second.
These are descriptive, not a causal turn-order experiment, but the direction
agrees with the deck’s need to evolve and attach twice.

The strongest confirmed behavioral signals were:

- Of 186 recovered Seek Inspiration source cards, 126 were Kyurem (67.7%), 31
  Conkeldurr (16.7%), 23 Annihilape (12.4%), and 6 Slowking (3.2%).
- Academy at Night produced a matching confirmed payload mix: 81 Kyurem, 18
  Conkeldurr, 12 Annihilape, and 11 Slowking among 131 confirmed placements.
  This directly supports the
  Academy → payload-on-top → Seek Inspiration sequence.
- Telepath Psychic Energy searched 139 Slowpoke, 52 Latias ex, and 36 Smoochum
  in 227 confirmed selections. Slowpoke was 61.2% of those targets.
- Poké Pad selected Slowking 79 times and Slowpoke 62 times in 192 confirmed
  selections. The evolution engine accounted for 73.4% of its targets.
- Of 865 confirmed manual attachments, 279 went to Mega Kangaskhan ex, 196 to
  Slowpoke, 136 to Latias ex, 117 to Smoochum, and 91 to Slowking. Only 14
  went to Kyurem. Energy is principally building the active engine or the next
  attacker; Kyurem is normally a copied-attack payload, not an attacker.
- Among 301 confirmed attack actions, 186 came from Slowking, 107 from
  Smoochum, seven from Slowpoke, and one from Kyurem. Delightful Kiss is a
  material early-game line, not a decorative attack.
- In 362 opening-Active choices that could be confirmed from subsequent state
  transitions, Mega Kangaskhan ex appeared 129 times, Smoochum 88, Slowpoke 63,
  Latias ex 48, Kyurem 20, Meowth ex 10, and Fezandipiti ex 4. These are availability-conditioned
  observations, not unconditional preference probabilities.

## Pilot plan

### Recovered high-confidence opening rule

The confirmed legal-option audit reveals a sharper opening heuristic than raw
frequencies alone: choose the first available card in this priority order:

1. Mega Kangaskhan ex;
2. Smoochum;
3. Latias ex;
4. Slowpoke;
5. Fezandipiti ex;
6. Meowth ex;
7. Kyurem.

This ranking reproduced 332 of 337 confirmed, nontrivial opening-Active prompts
(98.5%) across all three observed lists. It held in both turn orders; the bot's
later plan branches by turn order even though its opening-Active ranking mostly
does not. The five exceptions show that priority is not a complete strategic
model, so use the ordering as the default and still inspect the hand.

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
The 58.6% three-day final-list win rate is not a formal superiority result, and team names are
not normalized matchup labels. Use the highly repeated causal patterns—engine
development, final stacking, copied-attack distribution, Energy routing—as a
teacher. Do not turn raw action frequency into unconditional rules.

## Executable reverse-engineering audit

The research-only surrogate in
`poke_bot/slowking_reverse_engineered_policy.py` scores complete current legal
option sets and abstains when its causal rule is unsupported or low-margin. It
has an exact-zero serving-logit bypass. Its replay auditor verified all 768
games and required a subsequent state/log transition before treating a replay
choice as confirmed.

The current post-hoc audit covers 430 confirmed decisions and agrees on 410
(95.3%): opening Active 332/337, high-confidence Night Stretcher evolution
recovery 6/6, and Academy-to-Seek top-deck choice 72/87. On the 457
`ShumpeiNomura` games specifically it agrees on 226/228 covered decisions
(99.1%). These are deliberately selective coverage numbers, not whole-policy
accuracy, and the rules were refined using the aggregate corpus; the named
calendar splits in the receipt are diagnostic strata, not an untouched final
test.

The remaining Academy errors and all copied-attack target, discard-combination,
attachment, pivot, and long-horizon sequencing choices should be learned by the
option-conditioned neural policy and improved with exact-simulator search. They
should not be filled with lower-confidence hand rules.

## Sources paired into this guide

- Kaggle daily top-ladder archives for July 6–7 and July 30–August 4.
- Aggregate replay receipt,
  `state/slowking_multi_day_replay_distillation_2026-07-06_to_2026-08-04.json`,
  plus its eight checksum-bound daily receipts.
- Existing card-text and strategy synthesis,
  `docs/deck_guides/slowking-expert-brief.txt`.
- Existing causal/abstaining heuristic audit,
  `state/slowking_heuristic_research_v1.txt`.
- Executable surrogate audit,
  `state/slowking_reverse_engineered_policy_audit_v1.json`.
- Single-day precursor guide,
  `docs/deck_guides/slowking-shumpeinomura-distilled-guide-2026-08-04.md`.
