# RL Training Protocol

Schema version: `poke_bot.rl_training_protocol/v2`

This document is the authoritative human-readable workflow for specialist and
population reinforcement learning. The authoritative numerical values are in
`config/rl_protocol.yaml`; mutable execution state is in
`state/specialists.yaml`. If prose, configuration, and runtime behavior differ,
training must stop at the next safe boundary and the discrepancy must be
resolved without weakening a gate or rewriting historical evidence.

## 1. Shared deck-agnostic core

Derive one reusable deck-agnostic core from the strongest validated Alakazam
checkpoint. The source checkpoint, its checksum, the validation evidence, and
the derived-core checkpoint must be recorded in `state/specialists.yaml`.
Unknown paths or thresholds remain null until verified; they must never be
guessed.

Alakazam remains the active specialist through the complete iteration-30
training and evaluation boundary even if an earlier checkpoint passes. Core
derivation must not start early. After iteration 30, select the strongest exact
Alakazam checkpoint that passes both established gates (using LC55 when it has
passed, otherwise the separately authorized LC50-after-iteration-30 fallback).

The passed Alakazam checkpoint is a transfer-learning teacher, not the shared
core artifact itself. Create a write-once neutral transfer initialization that:

- preserves the card/state embeddings, spatial encoder, single temporal layer,
  option decoder, shared policy/value heads, and deck-agnostic auxiliary heads
  exactly;
- resets all matchup-adapter tensors to the deterministic zero-output roster;
- disables matchup routing and adapter training;
- discards specialist optimizer, scaler, scheduler, RNG, counters, and
  Alakazam-only training metadata; and
- records the exact source checkpoint/checksum plus a tensor-level preservation
  and reset audit.

Distill that neutral initialization for at most 25 supervised epochs on the
protected, balanced, multi-archetype top-ladder corpus. Use episode-disjoint
validation, patience 5, minimum improvement 0.0001, at least 500,000 decisions,
and a requested 12,288-decision batch. The Alakazam guide loss is zero during
core distillation. Train the shared policy, value, archetype, opponent-belief,
lethal-threat, and prize-race objectives. The zero-output matchup roster must
remain bit-identical throughout distillation.

Freeze the best validation checkpoint as the shared core only after verifying
its neutral identity, source provenance, balanced-corpus identity, transfer
compatibility, and zero specialist-adapter state. Subsequent specialists
hot-start from this frozen core and create/train their own specialist and
matchup heads; they do not inherit Alakazam's adapter weights.

### Versioned core refresh after Starmie

Starmie remains bound to the existing Alakazam-derived shared core for its
entire bootstrap and baseline-RL lifecycle. Do not restart, re-bootstrap, or
change Starmie's initialization to apply this refresh.

After Starmie becomes training-complete, the next-specialist handoff must build
a new versioned deck-agnostic core before bootstrapping the next unfinished
specialist. Every cumulative refresh uses every checksum-verified frozen
specialist available at that boundary as an immutable, inference-only teacher,
including Alakazam. No teacher may receive gradients, optimizer steps, replay
or rehearsal updates, parameter changes, re-freezing, or checkpoint
replacement.

Initialize the v2 student from core v1, distill balanced contributions from
both teachers over the protected balanced multi-archetype corpus, and train the
same shared policy, value, archetype, opponent-belief, lethal-threat, and
prize-race objectives used for core v1. Specialist matchup adapters remain the
deterministic zero-output roster and bit-identical throughout core
distillation. Never average specialist checkpoints or copy either teacher's
matchup adapters into the shared core.

The artifact is eligible for the next specialist only after a protected
checkpoint, checksum, reproducibility receipt, exact identities for every
teacher and the balanced corpus, neutral-adapter audit, and regression
evaluations against every teacher have been recorded. The core-transfer
contract runs exactly 80 inference-only games per teacher, requires at least
0.35 raw win rate against every teacher and at least 0.40 aggregate raw win
rate across all teachers. The 90% confidence intervals remain recorded as
diagnostics but do not block core transfer. These games are never training- or
replay-eligible. This transfer check is separate from, and does not alter, any
specialist baseline gate threshold. If validation is incomplete or fails,
the handoff fails closed before the next specialist's bootstrap. This does not
modify Starmie or invalidate core v1 and must not interrupt an active Starmie
run.

## 2. Sequential specialist training

Train exactly one active specialist at a time for every target deck/archetype.

### Pre-staged specialist handoffs

While the active specialist trains, the controller must pre-stage the next
executable specialist's dependency-free inputs. It selects against the same
canonical priority, frozen registry, corpus minimums, and accepted causal
router used by the boundary handoff; validates and checksum-binds the protected
expert corpus and exact 60-card representative; and may build the derived CPU
tensor pack on a staging host. The receipt is
`poke_bot.next_specialist_prestage/v1`.

Pre-staging is read-only to live production. It cannot update the active
specialist selector or runtime registry, materialize a successor gate, start or
stop a service, update a model, or claim that a missing asset is ready. Missing
corpus coverage, routing acceptance, representative decks, and CPU packs are
reported before the boundary so they can be completed without extending the
inter-deck pause.

Only work that depends on the exact newly passing checkpoint remains at the
boundary: freeze/register the passing specialist, distill and validate the
new cumulative core, run the exact 25-epoch hot-start from that core,
materialize the checksum-bound S+ gate, then atomically select and start the
managed specialist service. A pre-stage receipt never weakens or bypasses any
of those checks.

For each specialist:

1. Hot-start from the shared core.
2. Generate archetype, game-plan, and matchup policy heads following the
   validated Alakazam design. Matchup heads are required for every specialist,
   not only for Alakazam.
3. Before the first bootstrap or RL update, materialize the complete canonical
   matchup bank and enable the validated causal runtime router. This is the
   default launch state for Starmie and every later specialist; runtime-off is
   a fail-closed maintenance state, never a normal specialist launch mode.
4. Bootstrap on relevant expert top-ladder replays for exactly 25 supervised
   epochs.
5. Run baseline-phase RL iterations for that specialist. After every 5 RL
   epochs, run exactly 5 expert-replay rehearsal epochs.
6. Permit updates only to the active specialist.

Every specialist has a completed-iteration floor of 5 and a
completed-iteration ceiling of 15. A measured gate pass may transition at
iterations 5 through 15 inclusive. If the specialist reaches iteration 15
without a measured pass, freeze and register the exact iteration-15 checkpoint,
submit its required Kaggle copy (or persistently queue it if quota is
unavailable), and immediately begin the next unfinished specialist. Record
this outcome as `ceiling_accepted`, preserve the complete failed-gate result,
and never describe it as a measured gate pass. Ceiling acceptance is
training-complete and makes the immutable checkpoint eligible as an
inference-only public-mix opponent. Historical runs that completed under an
older bound retain their results as evidence, but those retired bounds are not
active policy and must not be copied into a new or resumed specialist launch.
For S+ opponents, the per-matchup win-rate floor is 30%; at most two S+
matchups may fall below that floor in one gate evaluation. The established
aggregate confidence and S-tier mean requirements remain unchanged.

After every completed iteration at or beyond the applicable gate floor, impose
a 30-second hard boundary pause before starting another collection. The exact
gate result must already be committed and visible during this pause. If the
terminal gate passed, commit its immutable marker and stop before any next
collection; otherwise resume only after the pause completes.

Historical core v2 used Hop's Trevenant and Starmie as its direct checkpoint
tensor sources. Beginning with the post-Lucario rebuild, every cumulative core
uses all checksum-verified frozen specialists accumulated so far, including
Alakazam, as equal-contribution immutable teachers.

Repeat the shared-core refresh before bootstrapping every later specialist.
Each refresh must include the newly passing specialist plus every other
checksum-verified frozen specialist, create a new versioned and checksummed
core, pass the established core acceptance checks, and only then hot-start the
next specialist from that latest core. A later specialist must not silently
reuse an older fixed core. Whether the final strongest core should replace the
bases of all completed specialists is intentionally undecided and requires a
separate explicit decision; this protocol does not perform that replacement.

The active specialist is never preempted. At each specialist handoff, choose
the next model only from specialists that do not yet have an exact frozen
checkpoint that passed both baseline gates after the iteration floor. First
prioritize specialists for which the registry has no existing specialist
model or checkpoint artifact. Within that
group, rank by descending current public-ladder meta inclusion share from the
pinned PTCG Ladder Meta snapshot. Only after that group is exhausted may the
scheduler select an unfinished specialist that already has an existing model
artifact; rank that second group by the same meta-share rule. An artifact may
establish availability without being protocol-valid or resumable: its recorded
restart policy still applies.

Refresh and pin the meta source immediately before a handoff; a refresh may
reorder only specialists that have not started. Equal shares use the stable
target-registry order. Missing or ambiguous source mappings remain null and
sort after verified shares within their availability group; they must not be
guessed. Public meta share is a training-priority signal only and never changes
a gate or supplies training examples.

A handoff may carry an explicit, recorded operator priority prefix or overlap
deferral. The named missing-model targets run first in their recorded order;
the remaining targets then return to the normal availability/meta ordering.
An overlap deferral may move a distinct missing-model target ahead of an
unfinished target substantially covered by the active or completed specialist
design. Neither mechanism removes a target, marks it complete, weakens a gate,
or permits an unrecorded reorder. The mutable tracker must name every promoted
target and the reason for the exception.

A completed specialist is immutable until every required specialist passes.
It must receive no gradients, optimizer steps, replay updates, rehearsal
updates, parameter changes, or checkpoint replacements. A frozen completed
specialist may serve only as an inference-only opponent in the eligible
public-mix pool. Its inference package must preserve its own checksum-pinned
causal matchup tree and adapter bank, so it remains routable while frozen.
For example, Starmie training uses Starmie as the sole trainable learner while
frozen Alakazam and frozen Hop's Trevenant may each route through their own
immutable inference-only matchup policy. The active learner's router and each
frozen opponent's router are separate identities; no optimizer may include a
frozen opponent's parameters.

Every saved bootstrap, rehearsal, RL-candidate, latest, passing, and frozen
checkpoint must carry the active specialist's canonical archetype ID and a
model ID containing the active run name. A specialist run must fail closed
before publication if a checkpoint inherits another specialist's label (for
example, an Alakazam label on a Trevenant checkpoint).

Matchup routing must use only causally available game-state observations. It
must not use opponent package identity, hidden simulator state, future actions,
or labels unavailable at submission time. A matchup head may update only from
sequences assigned to its relevant matchup; unknown or insufficient evidence
must fail closed to the specialist's base policy. Matchup heads freeze and
unfreeze with their owning specialist under the same immutability rules.

Every specialist deployment must materialize the complete canonical matchup
adapter roster in its checkpoint, including the specialist's own archetype.
An adapter with zero observations or zero learned weights is still a
materialized, routable slot; it must remain visible in monitoring and must not
be described as absent. This does not mean that every route contributes to
every decision. At each decision, the causal router selects at most one route,
and unknown or insufficient evidence uses the exact base-policy bypass.

### Staged v5 specialist and matchup-roster migration

At the next safe specialist handoff, replace the required v4 target roster
with the v5 logical roster in `state/matchup_adapter_roster_v5.json`. Retire
Raging Bolt, Gardevoir, N's Zoroark, Lopunny, and Cornerstone Ogerpon from
specialist selection and matchup routing. Rename the Festival Lead logical
target to Thwackey, matching the source deck's label, while retaining its
compatible physical adapter row. Append Team Rocket's Spidops as a distinct
future specialist and matchup route.

Existing v4 tensor rows are an immutable checkpoint-compatibility contract.
The five retired rows therefore remain as disabled physical tombstones: they
are never selected, routed, trained, or counted toward specialist completion,
and they receive no gradients. No existing row may be deleted or renumbered.
The existing `festival-lead` tensor row is exposed logically as `thwackey`;
this is a same-deck alias, not a new or reset tensor. The Team Rocket's Spidops
row is append-only and begins as an exact zero-output adapter. The v5 roster
may become active only after a causal router with the matching 23-row physical
class order passes validation, all legacy frozen checkpoints load without row
shifts, and fleet checksums agree.

After the current cumulative-core boundary, the explicit unfinished priority
prefix is Dragapult/Dusknoir, Dudunsparce, Marnie's Grimmsnarl ex, Cynthia's
Garchomp ex, Team Rocket's Mewtwo ex, Thwackey, and Team Rocket's Spidops.
Hammer-Pult then returns at its actual meta priority without the
existing-artifact priority penalty. Remaining unfinished v5 targets then
return to the established normal ordering.

The active specialist's own matchup route is mandatory for mirror self-play.
For example, a Hop's Trevenant specialist must contain the
`hops-trevenant` route, and causally recognized Hop's Trevenant mirror states
must be able to activate that route. A deployment must fail closed before
training starts unless:

1. the checkpoint's adapter roster exactly matches the canonical roster;
2. the active specialist's canonical archetype is present in that roster;
3. causal runtime routing is enabled and bound to a validated router receipt;
4. local and remote workers accept the same checkpoint, roster, router, and
   checksums; and
5. the first completed self-play collection receipt records a valid runtime
   audit and at least one activation of the active specialist's mirror route.

If the first self-play audit has no mirror-route activation, the completed
collection may be preserved for diagnosis, but no subsequent RL iteration may
start and the deployment must not be reported as routing-active. These
preflight and first-collection assertions apply to every current and future
specialist deployment, not only to Hop's Trevenant.

A specialist handoff must carry the router receipt, canonical roster checksum,
and runtime-enabled setting into the generated launch contract automatically.
An operator must not need to toggle routing after Starmie or any subsequent
specialist starts.

Dashboard status must distinguish three separate facts: a route slot is
materialized, routing runtime is enabled, and the route was observed in the
latest audited collection. The complete canonical roster remains listed even
when an individual route has zero observations. Zero observations must never
be rendered as evidence that the route slot is missing, and a materialized
zero-hit slot must never be rendered as a proven runtime activation.

## 3. Baseline-phase RL iteration

The current baseline-phase iteration contains exactly 8,192 new training
games:

- 7,168 public-mix games against public agents and eligible frozen
  specialists.
- 1,024 self-play games for the active specialist.

Distribute games as evenly as integer counts permit across applicable
opponents, archetypes, policy heads, and game plans. Balance the active
specialist going first and second as evenly as integer counts permit. Allocation
receipts must account for every game exactly once.

The 8,192-game setting is the current minimum, not a permanent ceiling. As the
frozen-specialist opponent pool grows, the training-game count may be raised
only by an explicit, recorded configuration change when the current budget no
longer supplies adequate opponent/head/game-plan coverage or variance. Any
increase preserves the 7/8 public-mix and 1/8 active-specialist self-play
shares, remains balanced by seat, and never counts research or holdout games
as training.

## 4. Research-only evaluations

Research evaluations are completely separate from training. The base program
totals 3,000 games; each frozen specialist adds 250 premium-holdout games:

- Official holdout: 1,000 games against 4 official agents, exactly 250 per
  agent, with 125 going first and 125 going second.
- Premium competition holdout: the established 2,000 games against 8 premium
  competition agents, plus exactly 250 games against every frozen completed
  specialist. Every opponent receives 125 games with the candidate going
  first and 125 going second. Frozen specialists are labeled `S+`, carry the
  established S-tier safety-floor requirement, and remain inference-only.

Lucario has one explicit future supersession rule. When our Lucario specialist
passes, is frozen, and is registered, all external premium-holdout opponents
whose canonical archetype is `lucario` are removed from every subsequent
premium holdout. The exact frozen Lucario specialist remains in the holdout as
an `S+` opponent. Historical results against the removed external Lucario
agents remain immutable and visible. This rule does not alter the separate
official research-control roster and never removes any non-Lucario opponent.

These games must never enter training datasets, replay buffers, rehearsal
data, expert corpora, advantage calculations, optimizer inputs, or any other
model-update path. They may be used only for public/competition gates,
checkpoint comparison, regression detection, and drift detection.

For every specialist, the exact 1,000-game official-four evaluation is a
required measured gate. Its aggregate and per-agent results must be preserved
and shown as regression/drift evidence. The premium competition holdout,
including all registered `S+` frozen specialists, is the second required
measured gate.

Use only the project's established gate thresholds and their authoritative
source artifacts. Never invent, infer, silently weaken, average away, or
replace a threshold. Until a threshold and source are verified, the
machine-readable value remains null and gate passage is impossible.

The stage-2 competition gate uses a skill-weighted 90% confidence-lower-bound
threshold of 0.50 for every specialist. A specialist may record earlier
research results, but it cannot complete or transition before its iteration-10
commit. The exact 250-game-per-opponent allocation, base eight-agent roster
plus every frozen specialist, both-seat balance, weighted-win-rate floor,
S-tier floor, individual-opponent floor, official non-regression requirement,
and audit requirement remain unchanged.

Lucario's completed iteration-9 gate decision remains checksum-pinned to the
0.25 individual-opponent floor under which that evaluation began. For
evaluations after that completed boundary—including Lucario iteration 10—the
operator-authorized individual-opponent floor is 0.15. This accommodates
structurally unfavorable matchups without changing the 0.50 skill-weighted
win-rate floor, the 0.50 90%-confidence lower-bound floor, the 0.40 S-tier mean
floor, the exact per-opponent allocation, or any audit requirement. If an
already-running controller completed a later exact evaluation using its
startup-pinned 0.25 contract, that original decision remains immutable; a
separate checksum-bound threshold-transition receipt may reclassify the same
exact result under 0.15. Training ledgers and checkpoint bytes must never be
rewritten for that reclassification. Every later specialist gate contract must
materialize 0.15 before training starts.

## 5. Passing a specialist and asynchronous Kaggle submission

When a specialist passes both the official-four gate and the measured premium
gate after the iteration-10 floor:

1. Freeze and register the exact passing checkpoint and checksum.
2. Record all reproducibility metadata and complete holdout results.
3. Mark the specialist training-complete. Kaggle submission completion is a
   separate asynchronous obligation and is not part of training completion.
4. Create one clearly labeled submission-copy record pinned to the checksum of
   that exact frozen passing checkpoint.
5. Submit that copy if the remaining five-submission daily quota permits and
   record its submission ID, timestamp, and returned score.
6. If quota is exhausted, mark the copy `pending`, append it to the
   persistent submission queue, and immediately begin or continue the next
   unfinished specialist.
7. Add the frozen checkpoint to the eligible public-mix inference-only pool
   and to every later premium holdout as an `S+` opponent.
8. Immediately begin the next unfinished specialist.

Before an upload, the submission builder and asynchronous queue processor must
both fail closed unless the package contains: (a) the exact frozen passing
checkpoint bytes and matching specialist archetype ID, and (b) the exact
60-card representative bound by checksum to that specialist run's measurement
deck contract, and (c) that specialist's validated checksum-pinned causal
matchup tree. The queued label, specialist ID, checkpoint checksum, deck-file
checksum, canonical card-list checksum, representative-registry checksum, and
matchup-tree checksum must remain immutable. A mismatched or stale
model/deck/router package is failed and must never reach Kaggle.

The submission queue is processed oldest-first whenever quota becomes
available, but every upload must also be spaced at least four hours after the
immediately preceding Kaggle submission. The processor reconciles the latest
submission timestamp, records the next eligible upload time, and leaves an
otherwise-ready copy pending until that time. This spacing wait is not a
failure and must not interrupt active specialist training. Queue work runs
without interrupting active specialist training. After Kaggle reports that the
daily limit has been reached, do not repeatedly retry during the exhausted
quota window. Every delayed copy remains pinned to its original frozen passing
checkpoint: delay must never cause retraining, modification, replacement, or
re-freezing of the specialist.

No completed specialist may be updated until every required specialist is
gate-passing frozen.

## 6. Population phase

After every required specialist is gate-passing frozen:

1. Unfreeze the population.
2. Begin local round-robin self-play against current and selected historical
   versions of our own models.
3. Continue the schedule of 5 RL epochs followed by exactly 5 expert-replay
   rehearsal epochs.
4. Treat all public, official, and premium agents as research-only benchmarks.
   None may supply population-phase training games.

The transition requires a complete roster audit, immutable passing-checkpoint
records, and explicit population-phase state in `state/specialists.yaml`.
Pending Kaggle copies do not prevent a specialist from being training-complete,
do not prevent all specialists from being marked baseline-passing, and do not
block this population-phase transition. Their immutable queue records continue
to be processed asynchronously.

## 7. Evidence and mutation rules

- `config/rl_protocol.yaml` locks the numerical protocol.
- `state/specialists.yaml` records facts, provenance, counters, checkpoints,
  holdouts, freezes, submissions, and unresolved evidence.
- Missing evidence is represented by null, never by a plausible substitute.
- Historical checkpoint checksums, holdout results, and submission records are
  append-only evidence.
- Any process that cannot prove research/training separation or frozen-model
  immutability must fail closed before an optimizer step.
