# Complete Fleet to Top Finish

Status: research execution plan  
Plan type: reusable `/plan`-style handoff  
Recorded: 2026-07-28  
Runtime authority: none

This document is a durable strategy and execution plan. It does not start,
stop, replace, or reconfigure training. `GOAL.md` and its typed canonical
sources remain authoritative for production behavior. Any future execution
that changes a production contract must first be recorded through the
`GOAL.md` design-change procedure and activated only at a safe receipt-backed
boundary.

## 1. Owner decisions captured by this plan

1. Complete the full planned specialist fleet.
2. Until the submission deck is explicitly selected, every specialist is a
   practice partner.
3. Do not designate, rank, or optimize any specialist as a finalist before deck
   selection.
4. A specialist's value before deck selection comes from the behaviors,
   matchups, and failure modes it contributes to the practice population, not
   from whether it appears individually submission-ready.
5. Once the deck is selected, use the completed fleet to harden candidate
   agents for that deck and select the final submission package.

These decisions refine how research interprets specialists. They do not change
the existing sequential-training, freeze, gate, or asynchronous submission
contracts.

## 2. Objective

Build a complete and behaviorally diverse specialist fleet, choose the
submission deck using fleet-wide evidence, and then use the entire fleet as an
anti-exploitability practice league to produce the strongest possible final
agent for that deck.

The desired outcome is not the largest number of individually high-scoring
specialists. The desired outcome is one submission deck and policy that:

- has a positive, statistically defensible edge against the strongest relevant
  field;
- has no catastrophic seat, matchup, or reliability failure;
- remains strong against counter-strategies and historical exploiters;
- fits the official submission resource and time limits; and
- is selected by locked local evidence rather than by one noisy public score.

## 3. Definitions

### Practice partner

Any checksum-bound, immutable specialist or historical checkpoint that is safe
for inference-only practice and adds useful opponent behavior. A practice
partner may be gate-passing or `ceiling_accepted`; its exact status must remain
visible.

### Training-complete specialist

A specialist that completed the canonical specialist process and was frozen
and registered under its true measured outcome. Training-complete does not mean
finalist.

### Audit submission

The automatic one-shot Kaggle copy required by `GOAL.md` revision 18. It is a
diagnostic observation of an immutable specialist package. It does not select a
submission deck, create a finalist, or authorize use of Kaggle results as
training data.

### Submission deck

The exact deck identity and legal 60-card representative selected by an
explicit owner decision after fleet evidence is reviewed.

### Finalist candidate

A candidate policy for the selected submission deck. This term must not be
used before the deck-selection decision.

### Release candidate

An immutable finalist checksum and package that has passed confirmation and is
eligible for the one-use locked release evaluation.

## 4. Strategic sequence

```text
Complete every specialist
        ↓
Freeze and characterize every practice partner
        ↓
Build fleet payoff and failure-mode maps
        ↓
Select the submission deck explicitly
        ↓
Train multiple candidate policies for that deck
        ↓
Use the complete fleet as a weighted practice league
        ↓
Confirm across seeds, seats, variants, and time
        ↓
Run one locked release evaluation
        ↓
Choose and freeze the final submission package
```

## 5. Phase A — Complete the specialist fleet

### Goal

Finish every required specialist under the existing sequential program while
preserving each completed specialist as an immutable practice partner.

### Required work

- [ ] Continue exactly one active specialist learner at a time.
- [ ] Keep the canonical 8,192-game baseline iteration unless the authoritative
      numerical protocol is explicitly changed.
- [ ] Preserve the iteration-5 through iteration-15 gate and ceiling behavior.
- [ ] Continue one-ahead preparation, guide research, exact representative
      binding, router validation, and terminal-path preflight.
- [ ] Freeze and register every training-complete specialist under its exact
      checkpoint, deck, guide, router, and gate identities.
- [ ] Preserve every failed gate and `ceiling_accepted` result without relabeling
      it as a measured pass.
- [ ] Continue cumulative-core attempts without allowing a rejected core to
      block specialist production.
- [ ] Treat automatic one-shot Kaggle copies as external observations, not
      finalist designations.

### Practice-partner dossier

Create or preserve the following evidence for every specialist:

- immutable specialist and checkpoint identity;
- exact deck representative and deck-family identity;
- gate status and iteration count;
- reliability, invalid-action, timeout, and valid-game rates;
- win rate split by opponent and seat;
- guide, router, and architecture identity;
- critical failure states and recurring decision errors;
- payoff vector against the practice fleet;
- behavior or matchup coverage not supplied by existing partners; and
- historical checkpoints worth retaining because their payoff vectors are
  meaningfully different.

### Full-fleet readiness gate

Phase A is complete only when:

- the canonical required-specialist set equals the set of terminal specialist
  dispositions exactly;
- every member is checksum-registered with its exact checkpoint, deck,
  matchup route, guide, gate, and terminal-transition receipts;
- every terminal disposition is explicitly recorded as a measured pass or a
  preserved `ceiling_accepted` outcome;
- no required specialist remains active, pending, duplicated, or unknown;
- all practice packages pass inference and reliability validation;
- evaluation/training isolation is proven; and
- the population contract agrees with the canonical member identities.

Pending asynchronous Kaggle observations do not block fleet completion. If
human protocol prose and the typed ceiling contract disagree about whether a
preserved ceiling outcome counts as training-complete, fail closed and
reconcile them before population activation.

### Phase A non-goals

- Do not choose a finalist.
- Do not stop the fleet because one specialist has a strong public score.
- Do not discard a weak but behaviorally distinct specialist.
- Do not spend finalist-scale tuning compute on every specialist.
- Do not use public Kaggle score as a training signal.

## 6. Phase B — Characterize the completed fleet

### Goal

Turn the collection of specialists into a measured practice population.

### Fleet evaluation matrix

Build a complete seat-balanced cross-play matrix containing:

- every current specialist;
- immutable original anchors;
- selected historical checkpoints with distinct payoff vectors;
- deck-list variants where one modal list does not represent the archetype;
- fixed official research controls for evaluation only; and
- external premium agents for evaluation only.

External agents and official controls remain training-ineligible.

### Measurements

For each practice partner, record:

1. Meta-weighted point win rate.
2. Macro-opponent win rate.
3. Separate going-first and going-second results.
4. Worst-decile and CVaR matchup performance.
5. Critical matchup floors.
6. Deck-variant dispersion.
7. Out-of-time performance decay.
8. Reliability and valid-game rate.
9. Payoff-vector novelty.
10. Best-response value against leading population mixtures.

Action accuracy, auxiliary-head accuracy, guide agreement, and public Kaggle
score are diagnostics, not primary strength metrics.

### Practice-partner roles

Assign one or more evidence-backed roles:

- **Anchor:** stable reference opponent.
- **Near-peer:** approximately even and useful for efficient learning.
- **Exploiter:** exposes a recurring weakness.
- **Counter:** targets a specific deck or strategy.
- **Tail specialist:** represents uncommon but strategically distinct play.
- **Historical guard:** prevents forgetting of an older behavior.
- **Reliability probe:** stresses long games, branching, or resource limits.

Roles affect sampling weight, not specialist value or permanence.

### Phase B completion gate

- [ ] Every practice partner has a complete payoff vector or a documented
      evidence gap.
- [ ] The matrix is balanced by seat and versioned by exact package checksum.
- [ ] Distinct deck variants are represented where material.
- [ ] The strongest exploiters and coverage gaps are identified.
- [ ] The matrix and metrics are frozen before deck selection begins.

## 7. Phase C — Select the submission deck

### Goal

Choose the deck only after the completed fleet makes its strengths, counters,
learnability, and robustness measurable.

### Candidate-deck evidence

For every serious deck candidate, evaluate:

- best currently attainable policy strength;
- robustness across common legal list variants;
- matchup distribution against the complete fleet;
- worst credible counters and recovery paths;
- first/second-seat sensitivity;
- draw and outcome variance;
- strategic ceiling under improved training;
- representation quality for its important decisions;
- availability and quality of expert demonstrations;
- search suitability, if search later becomes calibrated;
- runtime reliability and computational cost; and
- expected value against the projected final field.

Do not infer deck strength solely from the current specialist checkpoint. Deck
strength, policy maturity, and training-data quality are confounded and should
be reported separately where possible.

### Decision process

1. Freeze the candidate-deck list and evaluation protocol.
2. Evaluate every candidate against the same fleet snapshot.
3. Review aggregate value, lower-tail matchups, learnability, and reliability.
4. Identify the best deck and at least one credible alternative.
5. Record an explicit owner decision naming the exact submission deck.
6. Only after that decision, create finalist-candidate identities.

### Deck-selection gate

The selected deck should:

- lead or remain statistically competitive on meta-weighted value;
- avoid an unacceptable worst-decile matchup profile;
- have no unrepairable representation or engine-contract problem;
- have enough demonstrations or self-play signal to support focused training;
- fit the submission runtime comfortably; and
- have a credible path to improvement against its identified exploiters.

If no deck clears these conditions, extend evidence collection rather than
selecting based on a noisy public result.

## 8. Phase D — Build candidates for the selected deck

### Goal

Produce multiple independently trained policies for the selected deck and use
the complete fleet to reduce exploitability.

### Candidate population

Create at least:

- two independent training seeds using the best validated training recipe;
- one conservative candidate anchored to the strongest stable policy; and
- one experimental candidate incorporating only independently validated
  improvements.

Do not combine multiple unproven changes in one candidate.

### Technical experiment order

Run these experiment families in order:

1. **Selection proxy**
   - Calibrate the local field suite against timestamped historical Kaggle
     observations.
   - Require useful leave-one-specialist-out rank ordering before using the
     proxy for model selection.

2. **Learning signal**
   - Complete the predefined AWR beta, learning-rate, weight-cap, and
     replay-exposure study.
   - Compare row-balanced, game-balanced, and decision-stage-balanced credit.
   - Stop a branch when agreement rises across two evaluations without a
     corresponding field-strength improvement.

3. **Representation**
   - Measure and repair feature collisions for exact candidate ordinal,
     NUMBER, SKILL serial, duplicate attacks, maximum HP, evolution state,
     turn flags, typed energy, and remaining selection budgets.
   - Require mechanic-level metamorphic tests and an unseen/rare-card holdout.
   - Do not add strategic heads to compensate for missing inputs.

4. **Value and fusion**
   - Measure Brier score, ECE, calibration slope, and sibling-action ranking.
   - Run flat-versus-fused gameplay comparisons.
   - Zero or shuffle one head group at a time and retain a head only when it
     produces causal gameplay value.

5. **Belief model**
   - Compare uniform, frequency-preserving, and time-decayed anonymous priors.
   - Measure archetype/deck recall, per-card count calibration, support repair,
     and downstream gameplay effect.

6. **Search and distillation**
   - Keep deployment search disabled until value ranking and calibration pass.
   - Compare greedy against trusted belief search under identical checkpoints.
   - Test selective search on high-entropy or high-value decisions.
   - Distill trustworthy visit and root-Q targets into the greedy policy.
   - Prefer fast distilled greedy serving unless inference-time search has a
     positive lower confidence bound and no timeout regression.

### Scaling rule

Raise game volume only after an experiment shows repeatable improvement on the
confirmation panel. The target may grow toward millions of focused practice
games, but raw scale must never substitute for a validated learning signal.

If this chosen-deck hardening phase differs from the authoritative all-member
population contract, it remains a future design proposal until the owner
records and safely activates that change through `GOAL.md`.

## 9. Phase E — Weighted fleet practice

### Goal

Use every specialist as an available practice partner while allocating games
to the most informative opponents.

### Required population mechanics

- Complete seat-balanced payoff matrix.
- Immutable opponent snapshot within a generation.
- Current and selected historical practice partners.
- Meaningful historical selection based on payoff novelty.
- PFSP or equivalent near-peer sampling.
- PSRO or approximate meta-solver mixture.
- Hard-negative sampling from worst-LCB matchups.
- Anti-forgetting checks against anchors and historical guards.
- Atomic generation evaluation before replacing incumbents.

### Initial research mixture

Use this as a hypothesis to validate, not as an activated production setting:

- 12.5% mirror/self-play;
- 40% meta-weighted current fleet;
- 20% approximate Nash/PSRO mixture;
- 17.5% worst-LCB exploiters; and
- 10% uniform tail and historical guards.

All completed specialists remain in the eligible practice population even when
their current sampling weight is small.

### Promotion criteria

A candidate advances only when:

- its meta-weighted delta improves;
- both seat-specific deltas remain acceptable;
- worst-decile performance does not collapse;
- critical matchup floors pass;
- it does not forget declared anchors;
- the result repeats across independent training seeds; and
- reliability remains within submission limits.

## 10. Evaluation funnel

### Open development panel

Purpose: rapid iteration, not release evidence.

- 12–18 meta archetypes.
- Multiple deck variants for important archetypes.
- Fixed official controls and strong fleet partners.
- Approximately 1,200 games per checkpoint.
- Balanced seats and versioned opponent packages.

Advance when the pooled 90% lower confidence bound on
candidate-minus-incumbent is at least -2 percentage points, neither training
seed collapses, and no critical matchup regresses more than 5 points.

### Confirmation panel

Purpose: decide which hypotheses deserve release consideration.

- Three independent training seeds per arm.
- Approximately 4,000 field games per seed.
- Separate current-meta, frontier, and temporal holdouts.
- At least 500 games for important matchup claims.
- Predeclared comparisons with multiplicity control.

Advance when:

- the 95% hierarchical-bootstrap lower confidence bound on the meta-weighted
  delta exceeds zero; or
- the lower bound is no worse than -1 point while worst-decile performance
  improves by at least 3 points;
- neither seat-specific delta is worse than -2 points;
- no critical matchup regresses by more than 5 points; and
- valid-game rate is at least 99.5%.

The training seed is the top-level bootstrap unit. Do not claim paired random
games if the engine cannot expose identical randomness.

### Locked release panel

Purpose: one final unbiased release decision.

- One immutable candidate checksum.
- Approximately 8,000 games.
- Package-disjoint opponents.
- Unseen deck variants.
- Out-of-time data.
- Stable anchors plus a rotating frontier cohort.
- Balanced seats and production-equivalent inference.

Use a release suite once. If the candidate fails, burn that suite and version a
new future suite before additional tuning.

Release requires a positive 95% lower confidence bound on meta-weighted delta,
acceptable lower-tail performance, no critical collapse, and acceptable
reliability.

## 11. Final submission selection

### Decision rule

Select the final package from frozen local evidence. Kaggle is confirmation,
not the optimizer.

The final two active slots should represent one of these evidence-backed
strategies:

- two independent seeds of the selected deck;
- a stable main policy plus a validated experimental policy for the same deck;
  or
- the selected deck plus an explicitly approved complementary deck, if later
  owner strategy permits it and evidence strongly supports the hedge.

Do not churn the final two slots in response to short-term public-score
movement. Any change to the existing automatic specialist-copy policy for the
final window requires an explicit `GOAL.md` decision.

## 12. Known validity blockers to resolve before population results are trusted

These are research questions, not authorizations to modify code:

1. The canonical specialist target count and the population controller's
   expected member count must agree. Current evidence indicates an 18-versus-22
   mismatch.
2. Historical versions must actually be selected and accumulated; a field
   named `selected_history` is insufficient without state transitions.
3. The effective self-play fraction must match the canonical population
   contract. Current research found a possible default/configuration mismatch.
4. Development opponents and release opponents must be package-, deck-, and
   time-disjoint.
5. A `ceiling_accepted` partner must remain clearly distinguishable from a
   measured gate pass.
6. Population updates must not introduce sequential member advantage within
   what is presented as one generation.

Do not interpret a population experiment until these questions have
receipt-backed answers.

## 13. Stop conditions

Stop or redirect an experiment when:

- action or auxiliary accuracy rises across two evaluations but field strength
  does not;
- one training seed drives the entire apparent improvement;
- the improvement disappears under seat or deck-variant splits;
- worst-decile or critical-matchup performance collapses;
- invalid games or timeouts exceed the reliability budget;
- search cannot beat greedy with a positive lower confidence bound;
- shared-core distillation fails a declared per-teacher floor;
- a release suite has already influenced training decisions;
- an external evaluation agent enters training data; or
- the experiment requires changing the healthy active runtime without an
  explicit owner decision and safe boundary.

## 14. Decision records required

Future work should produce explicit, immutable records for:

- fleet completion;
- fleet payoff-matrix version;
- practice-partner roles;
- deck-candidate set;
- selected submission deck;
- candidate training recipes and independent seeds;
- confirmation-panel results;
- population mixture and historical selection;
- value/search calibration;
- locked release-suite identity;
- release decision; and
- final package and submission-slot decision.

## 15. Resume instructions for a later research task

1. Read `GOAL.md` completely.
2. Confirm that this plan has not been superseded by a later owner decision.
3. Read the typed canonical source for the phase being evaluated.
4. Refresh receipts before quoting mutable progress.
5. Identify the earliest incomplete checkbox whose dependencies are satisfied.
6. Keep analysis and recommendations read-only unless the owner separately
   authorizes implementation or production action.
7. Never treat this plan as authority to interrupt healthy training, alter the
   selector, submit an agent, or activate a staged design.

## 16. Compact plan checklist

- [ ] Complete every specialist.
- [ ] Preserve every specialist as an immutable practice partner.
- [ ] Build the full payoff matrix and failure taxonomy.
- [ ] Resolve population experimental-validity blockers.
- [ ] Freeze deck candidates and their evaluation protocol.
- [ ] Select the exact submission deck explicitly.
- [ ] Create multiple independently trained candidates for that deck.
- [ ] Validate learning-signal and representation improvements.
- [ ] Run weighted fleet practice with exploiters and historical guards.
- [ ] Confirm winners across seeds, seats, variants, and time.
- [ ] Run one locked release evaluation.
- [ ] Freeze the final package and protect the final submission slots.

## 17. Canonical references

- `GOAL.md`
- `docs/RL_TRAINING_PROTOCOL.md`
- `config/rl_protocol.yaml`
- `state/specialists.yaml`
- `state/matchup_adapter_roster.json`
- `ops/specialist_runtime_registry_v1.json`
- `ops/frozen_specialist_registry_v1.json`
- `ops/specialist_transition_graph.json`
- `ops/population_round_robin_v1.json`
- `docs/AWR_SHADOW_STUDY.md`
- `docs/STRATEGY_AUX_HEAD_ROADMAP.md`
- `docs/pokemon_rl_plateau_audit.ipynb`
- `submission/search_config.json`
