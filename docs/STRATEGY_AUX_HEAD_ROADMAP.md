# Strategy auxiliary-head roadmap

The purpose of an auxiliary head is to make the shared state representation
retain a strategically useful fact that the policy/value losses learn too
slowly.  A head is not justified merely because a quantity is measurable.

## Admission contract

Every candidate must satisfy all of these before it enters a production
checkpoint:

1. The label is causal at inference time, or is explicitly target-only
   privileged supervision that never becomes an input.
2. The label can be reproduced from the pinned engine/card library and replay
   observation without heuristic ambiguity.
3. Missing labels are masked, never silently replaced with zero.
4. A corpus audit reports coverage, class balance, per-archetype balance, and
   early/middle/late-game balance.
5. The head improves held-out policy/value metrics or matched-seed game
   results; parameter count alone is not evidence.
6. Warm-starting an older checkpoint is explicit and preserves every existing
   parameter exactly.

## Recommended order

### 1. Turn-resource memory

Predict four public engine flags from the temporal state:

- energy attachment already used this turn;
- supporter already used;
- retreat already used;
- stadium already played.

These flags are present in `observation.current`, but feature schema v5 does
not encode them directly.  Supervision therefore tests whether the temporal
layer remembers resource-consuming actions.  Use four independent BCE logits
and mask frames whose history was truncated before the current turn began.

This is the cheapest first experiment and is especially relevant to recovery
after Iono-style disruption.

### 2. Attack energy shortfall

Predict the minimum additional usable energy needed for the acting player's
best currently exposed attack: `0`, `1`, `2`, or `3+`.  The target must use the
pinned card library, typed attack costs, attached special-energy effects, and
the public active/bench board.  It must not infer a future top-deck.

This is more useful than reconstructing raw attached-energy count, which is
already present in board features.  A companion binary target may indicate
whether at least one attacker can be ready next turn after the normal
attachment allowance.

### 3. Evolution tempo

For each side, predict whether a strategically relevant evolution is
`available now`, `available next turn`, or `not represented in known
resources`.  The first version should use only public board plus the acting
player's own hand/deck identity.  Opponent-hand availability belongs only in a
privileged target head and must never be exposed as input.

### 4. Mobility / retreat readiness

Predict retreat-cost shortfall and whether a legal switch line is currently
represented in the visible action space.  This should be conditioned on
special conditions and attached energy rather than merely counting cards.

### 5. Public KO exposure

Predict whether either active Pokemon is within damage range of a legal
visible attack this turn and next turn.  This complements the existing
trajectory-derived `lethal_threat_head`: the existing label says a prize was
taken soon on the played line, whereas this head would be a public rules/card
calculation of tactical availability.

### 6. Hand actionability / disruption recovery

Predict a small ordinal target such as `dead`, `one productive line`, or
`multiple productive lines`, derived from the complete legal option set and
own-hand state.  Do not train a duplicate next-action classifier.  The useful
signal is whether the position retains alternatives after hand disruption.

### 7. Energy commitment quality

Predict whether the energy already committed to the board is `productive`,
`recoverable`, or `stranded`, plus the number of currently usable attackers.
The label must come from exact typed attack costs and legal retreat/switch lines,
not from card-name heuristics.  This is deliberately different from raw energy
count and from attack-energy shortfall: it asks whether past attachments still
support a plausible line after disruption or a forced switch.

### 8. Evolution-chain readiness

Predict the shortest legal turn distance to each board Pokemon's next useful
stage, capped at `3+`, and whether the chain is blocked by timing, a missing
pre-evolution, or a missing evolution card.  Own hidden cards may supervise the
acting player's target.  Opponent hidden cards may only be a privileged target
for representation learning and must never be copied into search state.

### 9. Damage and prize liability

Predict remaining effective hits-to-KO for both active Pokemon and the public
prize liability of the likely KO target (`1`, `2`, or `3`).  This should use
the pinned rules/card metadata for maximum HP, weakness, resistance, effects,
and special conditions.  It is complementary to the trajectory lethal target:
one describes exact public board geometry, the other says what happened on the
recorded line.

### 10. Opponent next-turn threat envelope

Use multi-label targets for public threats that can become legal next turn:
`attack`, `evolve`, `gust/switch`, `retreat`, `hand disruption`, and
`prize take`.  Start with an oracle target computed by bounded one-turn engine
enumeration.  Never label it from the eventual chosen opponent action alone,
which would confuse availability with player preference.

### 11. Board-development / bench quality

Predict a small ordinal description of setup quality: number of viable future
attackers, open bench capacity, and whether the current board has a legal
backup attacker.  This can help the shared trunk distinguish superficially
similar early positions without hard-coding an Alakazam-only sequence.

### 12. Deck-out and recovery horizon

Predict coarse turns-to-deck-out (`0`, `1`, `2-3`, `4+`) and whether a known
public/own-hand recovery line exists.  Use this only when the labels are
non-degenerate in the corpus; otherwise it belongs in exact search evaluation,
not in the neural loss.

### 13. Legal-line breadth

Predict capped counts of distinct productive action families available now and
after one legal action.  The target should collapse duplicate API candidates
by semantic effect and should not duplicate the policy's exact next-action
classification.  This is primarily a representation diagnostic for dead or
fragile hands.

### 14. Search-consistency targets (only when search is active)

Once bounded search produces trustworthy labels, add value-consistency and
policy-improvement heads: searched root value, best-line value gap, and policy
entropy over genuinely distinct legal actions.  These are later-stage targets;
self-play outcomes remain authoritative until simulator parity and search
calibration pass.

## Representation repairs before extra losses

The card-mechanics contract audit found state aliases that a new head should
not be asked to work around.  A schema migration should explicitly encode:

- exact bounded NUMBER/candidate ordinal and SKILL serial;
- maximum HP, stage/evolution stack, `appearThisTurn`, and once-per-turn flags;
- typed attack costs, prize liability, weakness/resistance, energy units,
  `ENERGY.count`, and remaining selection cost/counter budgets,
  preferably through a provenance-checked zero-gated metadata residual.

Fail-closed card/attack ID validation and official JSON-enum normalization are
now implemented without changing valid numeric feature rows. Cached datasets
from before enum normalization are schema-invalid and must be rebuilt.

After that migration, reconstruction heads for these fields are useful as
contract tests, but they should receive tiny weights or be disabled once the
encoder demonstrably preserves them.  Directly observable facts belong in the
state; heads are most valuable for future consequences, hidden-state beliefs,
and strategic abstractions.

## Proposed small experimental suite

Do not enable all candidates together.  The first isolated study should add
only three low-parameter heads to the frozen-base/no-head comparison:

1. four turn-resource-memory BCE logits;
2. four-class attack-energy shortfall plus ready-next-turn BCE;
3. three-class hand-actionability target.

If that suite passes offline representation metrics, unfreeze the trunk with a
small combined auxiliary-loss budget and run a matched-seed ablation for each
head.  A head that does not independently help Iono or pooled official-baseline
results is removed even if its own label accuracy is high.

The matchup-specific residual MLP proposal is a separate adapter experiment,
not another auxiliary head.  Its oracle/predicted routing evaluation should
remain isolated from this shared-trunk study.

## Candidates to reject or defer

- Raw attached-energy, public prize-count, or turn-number reconstruction: these
  are already observable and should be encoded directly.
- Opponent exact-hand targets used as policy inputs: privileged-label leakage.
- A classifier for the recorded next action: duplicates the policy objective
  and mistakes one sampled line for the only good line.
- Card-name-specific tactical heads: poor transfer and unnecessary parameter
  growth; use exact metadata and shared mechanics targets instead.
- Dozens of heads enabled at once: gradients become uninterpretable and make a
  warm-start regression impossible to attribute.

## Existing coverage

- `opponent_hand_head` and `opponent_remainder_head`: hidden-card beliefs.
- `lethal_threat_head`: near-term prize event on the played trajectory.
- `prize_race_head`: public prize-count state.
- `archetype_head`: opponent/deck identity supervision.
- Alakazam guide loss: deck-specific strategic action preferences.

The first new A/B should therefore be **turn-resource memory + attack energy
shortfall**, not another prize, archetype, or generic value head.

## Promotion evidence

The head suite remains experimental until it passes:

- at least 99% valid-label coverage on the authoritative temporal corpus;
- non-degenerate support in every data split;
- frozen-base/no-head control versus head-enabled training from the same
  checkpoint and replay split;
- exact matched-seed evaluation against Dragapult, Iono, Mega Abomasnow, and
  Mega Lucario;
- no per-baseline regression outside the predeclared confidence margin.
