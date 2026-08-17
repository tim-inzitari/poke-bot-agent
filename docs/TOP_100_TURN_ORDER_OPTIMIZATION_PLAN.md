# Top-100 Turn-Order Optimization Plan

Status: research execution plan

Plan type: standalone `/plan`-style companion

Recorded: 2026-07-29

Runtime authority: none

This is a separate companion to `docs/COMPLETE_FLEET_TOP_FINISH_PLAN.md`. It
does not modify or replace the complete-fleet plan. It focuses only on the
strategic opportunity created by top-ladder turn-choice behavior.

This plan does not authorize a runtime, selector, training, packaging, or
submission change. `GOAL.md` and its typed canonical sources remain
authoritative. Activating an always-second submission policy later requires an
explicit owner decision and safe receipt-backed activation.

## 1. Decision to inform

Determine whether the eventual submission should be built as an almost
fixed-second-seat deck-policy pair, then define how to select, train, evaluate,
and release that pair without overfitting to one aggregate observation.

Until the submission deck is selected:

- every specialist remains a practice partner;
- the full fleet continues to completion;
- no specialist is designated a finalist; and
- turn-order work is measurement and research only.

## 2. Observed top-100 behavior

The owner-provided aggregate reports that top-100 opponents chose second only
46 times in 46,661 choice events.

| Choice | Count | Share |
|---|---:|---:|
| First | 46,615 | 99.9014% |
| Second | 46 | 0.0986% |
| Total | 46,661 | 100% |

The naive 95% Wilson interval for the second-choice rate is approximately
0.0739%–0.1315%. Because games are clustered within opponent packages, the
final uncertainty estimate must bootstrap whole opponent versions and time
blocks rather than treating all games as independent.

### Current confidence

**Share with caveats.** The arithmetic is verified and the signal is too large
to ignore. Before it becomes a binding deck or release weight, its event grain,
time window, top-100 definition, choice-right ownership, deduplication, and
opponent concentration must be receipt-backed.

## 3. Seat-control calculation

Let:

- `c` = probability our agent receives the turn choice;
- `r` = probability the opponent chooses second when it receives the choice;
- `q = 1 - r` = probability the opponent chooses first; and
- our selected agent always chooses second when it receives the choice.

Then:

```text
P(we go second) = c + (1 - c) × q
P(we go first)  = (1 - c) × r
```

If choice ownership is fair (`c = 0.5`) and
`r = 46 / 46,661 = 0.0009858`:

```text
P(we go second) = 99.9507%
P(we go first)  = 0.0493%
```

That is approximately one forced-first game per 2,029 games. The naive
interval for our realized second-seat share is approximately
99.934%–99.963%.

If 46,661 instead represents all games rather than events where a top-100
opponent held the choice right, the direct realized second-seat estimate is
99.9014%. Resolve this denominator distinction before publishing or activating
the estimate.

## 4. Strategic conclusion

The observation does not establish that going second is generally better. It
shows that we can nearly control our seat if we choose second, because the
relevant opponent population almost always chooses first.

The actionable hypothesis is:

> Select a deck-policy pair with the highest absolute win rate playing second
> against first-optimized top-field agents, then choose second whenever
> offered.

Do not optimize for a large `WR(second) - WR(first)` difference by itself. A
deck that wins 45% second and 20% first has a large seat preference but remains
a weak submission. Optimize absolute second-seat strength and its lower
confidence bound.

For each opponent/deck/time stratum `s`, estimate:

- `π_s`: expected encounter weight;
- `c_s`: our probability of owning the choice;
- `r_s`: opponent probability of choosing second;
- `W2_s`: our win rate going second; and
- `W1_s`: our win rate going first.

The correct always-second ladder value is:

```text
L = Σ_s π_s × [
      (c_s + (1 - c_s) × (1 - r_s)) × W2_s
    + ((1 - c_s) × r_s) × W1_s
]
```

Do not multiply one global seat frequency by one global seat win rate.
Opponent choice, rank, deck, version, and matchup strength are correlated and
can otherwise create a misleading aggregate.

## 5. Phase A — Validate the observation

### Required event grain

Build the evidence table at one row per unique choice event with:

- replay/game ID;
- timestamp and observation window;
- explicit choice-right owner;
- explicit selected choice;
- resulting first/second seat;
- opponent entrant and submission version;
- opponent deck or inferred archetype;
- rank/rating at the time of the game;
- current-versus-historical top-100 membership;
- completion, timeout, and invalid-game state; and
- source receipt/checksum.

### Required quality checks

- [ ] Deduplicate replay retries and repeated ingestion.
- [ ] Confirm 46,661 counts eligible choice events.
- [ ] Confirm the 46 numerator means explicit second choices.
- [ ] Measure missing, unknown, censored, and invalid choice events.
- [ ] Verify choice-right ownership is approximately 50/50.
- [ ] Freeze top-100 membership at match time rather than applying today's rank
      retrospectively.
- [ ] Split different versions of the same entrant.
- [ ] Report encounter-weighted and equal-agent rates.
- [ ] Bootstrap opponent versions and time blocks for a cluster-aware interval.
- [ ] Report maximum agent share, top-five share, concentration, and effective
      opponent count.
- [ ] Split ranks 1–10, 11–25, 26–50, and 51–100.
- [ ] Split by deck family and recent time window.
- [ ] Attribute the 46 second-choice exceptions to exact packages where
      possible.

### Why the 46 exceptions matter

If most exceptions come from one or two deterministic packages, they are not
binomial noise. They are identifiable counter-strategies that can force our
always-second agent into its rare first-seat state. Preserve those packages as
explicit practice partners and first-seat safety probes.

### Validation gate

The observation becomes strategically binding only when:

- denominator and choice semantics are exact;
- choice-right ownership is verified;
- the cluster-aware estimate remains extremely first-preferring;
- the recent seven-day and 10,000-event windows show no material reversal;
- no important rank or deck stratum contains an unmodeled second-preferring
  subgroup; and
- the 46 exceptions are attributed or explicitly categorized as unknown.

## 6. Phase B — Build the seat-conditional fleet matrix

### Goal

Keep completing the fleet while making every practice partner useful for
turn-order analysis.

### Required matrix cells

For every candidate deck against every practice partner, preserve separately:

1. Candidate second / partner first.
2. Candidate first / partner second.
3. Natural choice behavior.
4. Candidate second against the exact first-choice strategy used by that
   partner on ladder.
5. Candidate first against exception packages that prefer second.

Never collapse these cells before deck selection.

### Per-specialist turn-order dossier

- Natural first/second choice counts.
- Choice rate by opponent and date.
- `WR(first)` and `WR(second)`.
- `WR(second versus partner-first)`.
- Reliability, timeout, and invalid-action rate by seat.
- Critical first-seat and second-seat failure states.
- Payoff-vector novelty within each seat.

### Required first/second win-rate views

Every view must use the candidate's **actual engine-recorded seat**. Never infer
seat from choice ownership, requested policy, or which package normally
prefers first.

At one row per unique scheduled game, preserve:

- candidate deck, checkpoint, package, training seed, and matchup-adapter
  digests;
- exact opponent entrant, version, deck or inferred archetype, and rank at
  match time;
- rules and engine version;
- source (`natural_ladder`, `forced_evaluation`, or `adaptive_training`);
- choice-right owner, explicit choice, assigned seat, and actual seat;
- scheduled, started, valid, and training-consumed state;
- win, loss, draw, candidate fault, opponent fault, neutral infrastructure
  failure, or cancellation-before-start; and
- timestamp, evaluation block, and replay/game identity.

For each actual seat `j ∈ {first, second}`, report:

```text
N_valid,j = wins_j + losses_j + draws_j
WR_j      = wins_j / N_valid,j
PWR_j     = (wins_j + 0.5 × draws_j) / N_valid,j
VGR_j     = N_valid,j / N_started,j
WY_j      = wins_j / N_started,j
```

Do not show a percentage without its numerator and denominator. Candidate
timeouts, invalid actions, and crashes remain visible through candidate-fault
rate and win yield even when a symmetric predeclared competitive-validity rule
excludes them from `WR`.

The review surface must show simultaneously:

1. An actual-seat overview with scheduled, started, valid, W/L/D, fault counts,
   `WR`, `PWR`, 95% interval, `VGR`, and win yield.
2. A fixed-mix comparison using the same frozen opponent weights in the first
   and second columns.
3. Encounter-weighted micro, equal-opponent-version macro, top-100-weighted,
   and worst-decile results, each labeled by weighting method.
4. An opponent-version matrix with first and second cells shown side by side.
5. Choice owner × explicit choice × actual seat × result reconciliation.
6. Rank-at-match-time and rolling-time views split by actual seat.
7. Assigned, actual, and training-consumed seat ratios.

Raw single-cell displays may use 95% Wilson intervals. Ranking and promotion
claims require a hierarchical or block bootstrap that resamples training seed,
exact opponent version, and whole schedule/time blocks while recomputing frozen
target weights. Use paired resampling only when the schedules really are
paired. Mark cells with fewer than 100 valid games or fewer than 10 effective
opponent-version clusters as `LOW_PRECISION`; retain the existing minimum of
500 games for important matchup claims.

Natural-seat `WR(second) - WR(first)` is observational because opponent
strength, deck, version, and choice behavior affect the realized seat. Treat a
seat delta as causal only on a predeclared forced-seat, opponent-balanced
schedule. Never pool forced evaluation, natural ladder, and adaptive-training
games; never condition only on completed games; and never mix candidate,
opponent, or ruleset versions silently.

### Informative contrast candidates

Do not preselect a deck, but preserve contrasting hypotheses:

- second-preferring or turn-one-pressure decks;
- conventional first-preferring decks;
- decks with small seat sensitivity but strong absolute value; and
- the exact packages responsible for the 46 second choices.

Hammer-Pult is an informative second-seat hypothesis because its current guide
describes a going-second turn-one pressure line. Archaludon and Rocket Spidops
are useful first-preferring contrasts. These are falsification candidates, not
finalist designations.

## 7. Phase C — Select the deck-policy pair

### Primary ranking metric

Rank candidates first by:

```text
LCB[WR(candidate second versus top-field opponent first)]
```

Then consider:

- top-100 ladder-weighted expected win rate;
- broad-ladder expected win rate;
- macro-opponent second-seat win rate;
- worst-decile second-seat matchup;
- performance against the 46 exception packages;
- first-seat legality, reliability, and minimum strength;
- sensitivity to a changing choice meta; and
- submission runtime and packaging reliability.

Balanced win rate remains a diagnostic. It must not be the primary deck ranker
when the final seat distribution is approximately 99.95% second.

### Deck-selection gate

Select an always-second deck-policy pair only when:

- its absolute second-seat lower confidence bound beats the alternatives;
- the result repeats across independent policy seeds;
- the gain is not concentrated in a few weak opponents;
- its worst important top-field matchup is acceptable;
- first-seat behavior remains legal, reliable, and above a declared floor;
- it remains competitive under plausible turn-choice shifts; and
- the exact turn-choice action is supported and attestable in the final
  package.

If no deck has a real absolute second-seat edge, do not choose second merely
because the seat is controllable.

## 8. Phase D — Train after deck selection

### Seat allocation

Use one exact research curriculum:

- **7/8 primary stream:** candidate second, practice partner first.
- **1/8 guard stream:** candidate first, practice partner second, with extra
  opponent weight on packages responsible for the observed exceptions.

This is an exact `candidate-first : candidate-second = 1 : 7` allocation, or
12.5% first and 87.5% second. Enforce it in deterministic eight-game quota
units for each candidate × training seed × frozen opponent-version block: one
assigned-first job and seven assigned-second jobs.

Record assigned, actual, and training-consumed seat separately. A failed or
invalid slot remains in reliability accounting and may be replaced only by the
same seat/opponent/version under a predeclared rule. Training ingestion must
preserve 1:7 exactly or block rather than silently drift.

The 1:7 curriculum is not an estimate of ladder seat frequency. Confirmation
evaluation remains deliberately stratified for inference and is analytically
reweighted to the projected ladder distribution afterward.

### Opponent allocation

Treat opponent selection and seat selection as independent controls:

- preserve the full completed fleet as eligible practice partners;
- use PFSP/PSRO or equivalent weighting among first-playing partners;
- emphasize near-even and worst-LCB matchups while the candidate is second;
- retain historical guards to prevent forgetting;
- include the identified second-choice exception packages in the first-seat
  guard stream; and
- freeze the opponent snapshot within each experimental generation.

### Learning priorities

Mine and rehearse:

- candidate-second losses against first-optimized opponents;
- opening sequences and turn-one decision branches;
- prize/resource plans that depend on seat;
- failure states unique to the opponent receiving its preferred first seat;
- rare candidate-first failures against second-preferring exceptions; and
- seat-specific calibration errors in value and outcome heads.

## 9. Phase E — Evaluate efficiently

### Do not sample the natural seat ratio

At a 99.9507% second-seat rate, 10,000 naturally distributed games would
produce only about five first-seat games. That is insufficient to measure the
fallback.

Instead, run stratified evaluation and analytically reweight the strata.

### Development allocation

Use approximately:

- 80% forced candidate-second games; and
- 20% forced candidate-first safety and exception games.

Use common opponent/deck/seat schedules between candidates when the engine
supports them. Stop losing branches with predeclared sequential confidence
bounds.

### Confirmation allocation

Use approximately:

- 90% candidate-second games against first-optimized top-field partners; and
- 10% forced-first guard games, concentrated on exception packages and diverse
  archetypes.

Report:

1. Absolute second-seat win rate and confidence interval.
2. Candidate-minus-incumbent second-seat delta.
3. Top-100 ladder-weighted expected win rate.
4. Broad-ladder ladder-weighted expected win rate.
5. Macro-opponent and worst-matchup results.
6. First-seat reliability and strength floor.
7. Natural-choice action-path audit.
8. The complete actual-seat overview and fixed-mix first/second comparison.
9. Assigned-versus-actual seat mismatch and seat-specific fault rates.

The optional curriculum diagnostic is:

```text
WR_trainmix = (1 / 8) × WR_first,std + (7 / 8) × WR_second,std
```

Label this only as the 1:7 training-mixture value. It is not the natural-ladder
projection and must not replace the separately calculated always-second value.

### Sensitivity grid

Evaluate opponent second-choice rates of:

- 0.1%;
- 1%;
- 5%;
- 10%;
- 25%;
- 50%; and
- 100%.

With fair choice ownership, these force our always-second agent first in about
0.05%, 0.5%, 2.5%, 5%, 12.5%, 25%, and 50% of games respectively.

For every pair of candidate decks, calculate the opponent-choice rate at which
their ordering reverses. This breakpoint—not an arbitrary percentage—is the
monitoring threshold that should trigger reconsideration.

## 10. Phase F — Release gate

The locked release package must bind:

- exact deck checksum;
- exact policy checkpoint;
- deterministic `second_if_allowed` choice policy;
- matchup router and guide identities;
- native-choice action-path test;
- submission resource and timeout receipt;
- top-field choice-distribution receipt;
- stratified evaluation schedule; and
- analytically reweighted release result.

Release requires:

- positive lower confidence bound versus the selected incumbent;
- best or statistically competitive absolute second-seat value;
- no critical top-field collapse;
- a passing first-seat correctness and reliability floor;
- acceptable performance at the predeclared choice-meta breakpoint; and
- exact package behavior matching the evaluated choice policy.

The current repository policy defaults Kaggle packages to
`first_if_allowed`. Packaging support for `second_if_allowed` exists, but
activating it for the eventual deck requires an explicit owner-level contract
change and safe-boundary receipt. Never silently inherit or flip the default.

## 11. Stop conditions

Stop or redirect the always-second strategy when:

- the 46/46,661 denominator or event semantics are wrong;
- a cluster-aware analysis shows the behavior is not broad;
- recent top-field choice behavior crosses a candidate-ordering breakpoint;
- second-seat strength is only a relative improvement and remains absolutely
  weak;
- the result is driven by a few low-value opponents;
- exception packages expose a catastrophic first-seat failure;
- candidate-second training damages reliability or core mechanics;
- the native packaged choice differs from the evaluated choice; or
- the strategy requires changing healthy production without explicit
  authorization and a safe boundary.

## 12. Decision records required

- Turn-choice source and data-quality receipt.
- Top-100 membership and observation-window receipt.
- Encounter-weighted, equal-agent, and cluster-aware estimates.
- Exception-package registry.
- Seat-conditional fleet payoff matrix.
- Deck-policy candidate set.
- Seat-allocation experiment results.
- Choice-meta sensitivity and reversal breakpoints.
- Exact owner deck and turn-choice decision.
- Locked release and package-behavior receipt.

## 13. Compact execution checklist

- [ ] Validate the 46/46,661 event grain.
- [ ] Verify fair choice ownership.
- [ ] Attribute the 46 exceptions.
- [ ] Produce concentration-aware choice estimates.
- [ ] Build the seat-conditional fleet matrix.
- [ ] Rank decks by absolute second-seat strength against top-field-first.
- [ ] Preserve first-seat correctness as a hard floor.
- [ ] Select the exact deck-policy pair explicitly.
- [ ] Enforce exact 1:7 candidate-first:candidate-second training quotas.
- [ ] Reconcile assigned, actual, and training-consumed seat ratios.
- [ ] Publish actual-seat and fixed-mix first/second win-rate views.
- [ ] Train against the full fleet with candidate second in the primary stream.
- [ ] Evaluate stratified seats and reweight analytically.
- [ ] Stress-test turn-choice meta shifts and calculate reversal breakpoints.
- [ ] Bind `second_if_allowed` into the locked package only after authorization.
- [ ] Pass the locked release gate.

## 14. Source references

- `GOAL.md`
- `docs/COMPLETE_FLEET_TOP_FINISH_PLAN.md`
- `docs/FINAL_MODEL_CAPACITY_AND_DECISION_REPLAY_PLAN.md`
- `config/rl_protocol.yaml`
- `ops/kaggle_submission_policy.json`
- `config/specialist_runtime.env`
- `docs/RL_TRAINING_PROTOCOL.md`
- `state/specialists.yaml`
- `docs/deck_guides/`
