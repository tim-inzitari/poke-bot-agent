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

If the bounded parameter-space initialization set produces four valid
gameplay-regression failures, one final architecture-repair attempt is
permitted. It must evaluate each immutable frozen teacher only on expert games
whose acting archetype matches that teacher, using the exact causal
board/action history and legal options, and train against that teacher's
greedy action with the machine-locked loss weight. Rows without a matching
teacher remain masked from this additional objective and retain their ordinary
expert targets. This is behavior-level distillation, not another seed retry:
teachers remain gradient-free and immutable, the protected balanced corpus and
exact 25-epoch schedule remain unchanged, and the established per-teacher and
aggregate gameplay gates may not be weakened. No further pass-seeking attempt
at that same boundary is authorized if this single behavior-repair candidate
fails.

The artifact is eligible for the next specialist only after a protected
checkpoint, checksum, reproducibility receipt, exact identities for every
teacher and the balanced corpus, neutral-adapter audit, and regression
evaluations against every teacher have been recorded. The core-transfer
contract runs exactly 80 inference-only games per teacher, requires at least
0.35 raw win rate against every teacher and at least 0.40 aggregate raw win
rate across all teachers. The 90% confidence intervals remain recorded as
diagnostics but do not block core transfer. These games are never training- or
replay-eligible. This transfer check is separate from, and does not alter, any
specialist baseline gate threshold. A failed candidate remains rejected,
immutable, and ineligible to initialize a specialist. It does not stop
production: the handoff immediately selects the latest checksum-accepted
cumulative core and hot-starts the next specialist from that accepted
fallback. Cumulative core V9 is the current accepted fallback. The
post-Thwackey V10 candidate is preserved as a rejected gameplay-regression
attempt and is not bootstrap-eligible. Matchup Adapter V6 is a separate
checkpoint-format version and must never be displayed or interpreted as the
current cumulative-core generation. The controller still attempts a new
cumulative refresh from all frozen teachers after every later completed
specialist.

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

Every specialist transition must be clean and automatic. Before the managed
trainer is allowed to start, its preflight must validate both the selected
training-launch path and the exact terminal freeze, package, asynchronous
submission, and next-specialist handoff path. This includes an exact
checksum-bound 60-card representative stored under the specialist's logical
ID; an alias alone is not sufficient terminal evidence. A deterministic
missing or malformed transition input therefore fails before RL wall time is
spent.

A successful terminal trainer exit directly starts the idempotent gate
handler. That handler freezes and registers the exact passing checkpoint,
authorizes or queues its non-blocking Kaggle copy, and starts the managed
handoff. The periodic gate supervisor is recovery-only and must never be the
normal transition mechanism. Replaying either handler is safe only when its
recorded identities are unchanged.

### Current-deck guide north star

Preparation of every new specialist must research and checksum-bind one
specialist-specific current-deck guide contract before bootstrap. The guide
follows the validated Alakazam pattern: a sparse teacher ranks the complete
legal action stage only when cited strategy evidence and the causally
observable public state support a high-confidence preference. The resulting
masked, confidence-weighted loss shapes the existing shared policy logits. It
is not a separate serving action path, may not inspect hidden or future state,
and never overrides the authoritative fused policy.

The guide is temporary scaffolding. Its loss weight ramps from 0.01 to 0.05
during bootstrap epochs 1–5 and remains at no more than 0.05 through the exact
25-epoch bootstrap. After bootstrap, a separate training-ineligible,
replay-ineligible evaluation estimates the association between guide/policy
agreement and wins, both overall and by matchup. The weight remains fixed only
while the lower confidence bound of that win-agreement lift is positive. Two
consecutive non-positive evaluations multiply the weight by 0.8; values below
0.005 become zero. The weight may never rise above its bootstrap maximum.
Every change requires a checksum-bound schedule receipt.

This ramp/hold/anneal lifecycle is the owner's protected goal-path guide
vision. It may not be silently removed, replaced with a permanently fixed
weight, or driven by ordinary training outcomes. It remains a required part of
the guide contract alongside the dedicated two-task research workflow.

Training games and formal gate games never tune the guide weight. Missing,
ambiguous, or partially scored legal stages are masked, not assigned a false
target. Each checkpoint reports guide rows, loss, policy agreement, win and
non-win agreement, agreement lift, and current weight. The one machine-readable
authority for this schedule is
`config/rl_protocol.yaml#/specialist_training/current_deck_guide`; individual
researched contracts live under `config/deck_guides/`.

Pre-stage readiness is bound to the expert corpus that bootstrap will actually
open. The guide contract's row count, specialist identity, and guide version
must match a `poke_bot.current_deck_guide_corpus_ready/v1` receipt beside that
selected protected pointer. The receipt, pointer, and selected manifest must
agree on their SHA-256 identities, decision count, and a nonzero guide-row
count. Every daily shard is checksum-validated before the guide corpus is
atomically promoted. The derived CPU pack is not built until this binding is
ready. A separate guide artifact, an unrelated unguided corpus, or a YAML-only
row claim can never satisfy pre-stage.

Every researched deck guide also requires a shareable expert brief under
`docs/deck_guides/`. It must identify the specialist and guide contract, cite
the same strategy-source set, explain the proposed principles, abstention
conditions, safety limits, and open review questions, and contain no more than
10,000 words. The guide contract records the brief's path, SHA-256 checksum,
and exact word count. A missing, oversized, identity-mismatched, source-
mismatched, or checksum-mismatched brief blocks successor pre-stage readiness.
The intended reviewers are world-champion and equivalent Pokémon TCG
subject-matter experts.

The write-up is always a practical guide for a human piloting the deck, not a
report about the training system. Its main body covers strategic identity,
variants and card roles, setup, going-first and going-second plans,
turn-by-turn sequencing, resource and attack planning, bench and prize
management, matchup plans, recovery lines, common mistakes, and decision
checklists whenever the reviewed evidence supports them. Uncertain or
format-dated advice is labeled. Exact mechanics are never invented.
Heuristic-extraction and training-audit details may appear only in a short
appendix.

The research and heuristic-extraction work for each specialist is assigned to
one dedicated subagent using the highest reasoning capability available in the
active environment. That subagent has exactly two tasks: produce the
expert-facing guide and extract its causal, abstaining heuristics. It may not
operate production training, selectors, services, dashboards, or unrelated
architecture. The production controller validates and integrates the returned
artifacts.

A specialist declared `nonlinear` requires more than a sparse guide. Before
pre-stage can be ready, a checksum-bound
`poke_bot.nonlinear_specialist_decision_support/v1` contract and validation
receipt must bind the complete 17-input fused policy to every declared branch
system, give each system an exact causal input set and a training-ineligible
scenario gate, preserve mask-not-zero behavior, and require the exact terminal
runtime gate. A pre-existing pre-stage receipt that lacks this binding is
superseded and must be reissued.

Hammer-Pult is nonlinear. Its required systems cover branching setup and pivot
choice, attacker and Drakloak-engine preservation, typed-Energy/resource
planning, Phantom Dive target and prize routing, disruption timing under both
possible Crushing Hammer outcomes, bench/recovery routing, and guide-weight
annealing. The sparse Hammer teacher remains deliberately narrow and causal;
the learned fused policy owns decisions outside that safe scaffold.

For each specialist:

1. Hot-start from the shared core.
2. Generate archetype, game-plan, and matchup policy heads plus the
   checksum-bound current-deck guide objective following the validated Alakazam
   design. Matchup heads are required for every specialist, not only for
   Alakazam.
3. Before the first bootstrap or RL update, materialize the complete canonical
   matchup bank and enable the validated causal runtime router. This is the
   default launch state for Starmie and every later specialist; runtime-off is
   a fail-closed maintenance state, never a normal specialist launch mode.
4. Bootstrap on relevant expert top-ladder replays for exactly 25 supervised
   epochs.
5. Run baseline-phase RL iterations for that specialist. After every 5 RL
   epochs, run exactly 5 expert-replay rehearsal epochs.
6. Permit updates only to the active specialist.

### Expanded strategic heads

The current corrected Dudunsparce learner and every subsequent cumulative core
and specialist contain the complete expanded strategic-head architecture. No
frozen V5 checkpoint is rewritten. During each ordinary full-model RL epoch and
each scheduled expert-rehearsal epoch, every architecture-present head with
valid exact causal labels participates in the loss and receives gradients.
Rows without a valid target are masked and contribute neither a fabricated
zero target nor a gradient.

Training activation is distinct from inference activation in checkpoints
created before owner decision 16. The currently executing iteration retains
its checksum-pinned flat-policy path, but the next safe-boundary runtime must
replace it with the canonical learned decision-fusion path. That path consumes
value, archetype, opponent-hand, opponent-remainder, lethal, prize-race, and
all eleven expanded strategic heads. Matchup adapters remain gated by the
causal router, and an absent current-deck guide is an exact bypass.

Implementation and shadow-training status is
`active_training_runtime_shadow`, not serving-path activation.
The checksum-contract validation receipt is
`state/expanded_strategic_heads_validation_v1.json`. It binds the complete
11-head inventory, target schema
`poke_bot.expanded_strategic_targets/v2`, target digest
`sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f`,
schedule schema `poke_bot.expanded_strategic_schedule/v1`, and schedule digest
`sha256:e471f58915df0cbe88b837de6fbe532e6416aa028a538b92a11ec788621f45dc`.
The earlier validation receipt proves architecture, labels, losses, gradients,
persistence, and handoff behavior; it does not by itself authorize the fused
serving path. Activation additionally requires a checksum-bound fusion receipt
proving deterministic local/remote logits, nonzero influence from every
required head, causal information use, and acceptable throughput and memory.
Once activated, the fused path is mandatory for Dudunsparce and every successor
specialist; silently omitting a required architecture-present head fails
closed.

Cross-platform local/remote parity means float32 logits within the canonical
absolute tolerance and exactly identical greedy decisions; it does not require
byte-identical floating-point results from different CPU kernels. Serving
acceptance is owned by
`config/rl_protocol.yaml#/specialist_training/decision_fusion/activation/performance_acceptance`.
The relative flat-versus-fused microbenchmark is diagnostic because activating
the required heads necessarily adds work. The serving checkpoint must still
meet the Blackwell absolute decision-throughput floor, memory ceiling, no-OOM
rule, and its own exact current premium evaluation plus 1,000-game official
evaluation. The premium count is 250 times the three active external opponents
plus every frozen specialist registered when that child is evaluated.

Target masking still applies to each head's direct auxiliary loss. A missing
direct label contributes zero direct head loss and is never replaced by a
fabricated target. Because the head output is part of the learned action path
after fusion activation, ordinary policy loss may still backpropagate through
that output; this is joint policy learning, not an inferred auxiliary label.

The action decoder supplies a selected-action Q estimate plus separate
legal-candidate scorers for action type, target binding, and
resource/source/tool/energy binding. Their labels come only from the canonical
factorized action stages and decoder binding contract. The selected candidate
may train against the terminal Monte Carlo return; unselected candidates remain
masked unless a checksum-bound audited search or counterfactual target proves
their value. A separate action-utility head predicts immediate damage, cards
drawn, energy change, open-bench change, prize change, and knockout. Utility
labels require an exact immediate post-action transition and are masked when
that transition is unavailable.

State-level heads predict tactical outcomes over the next one, two, and three
same-seat decision frames; the opponent response after the selected action and
before the same seat next acts; next-decision hand, deck, attached-energy,
bench-space, attachment-availability, and retreat-availability resources; game
phase; win/draw/loss outcome distribution; and log-one-plus remaining complete
game turns. Tactical, response, and resource components use independent masks.
Incomplete, terminal, truncated, ambiguous, or unavailable targets are absent,
not numeric zero.

Game phase is deterministic, public-state-only, and uses this precedence:
`closeout` when our prizes are at most one or the existing exact lethal label
is positive; `prize_race` when both players have at most three prizes;
`stabilize` when we trail by at least two prizes; `setup` during our first two
turns; otherwise `pressure`. Missing required fields mask the phase row.

All expanded labels are training-only. They are derived on the complete
trajectory before any context truncation, carry a versioned target schema and
provenance digest, and are never placed in board, option, history, routing, or
serving inputs. A malformed present label fails closed before an optimizer
step.

The exact 25 supervised bootstrap epochs use one cumulative schedule:

- epochs 1–5 train action Q, action-type, action-target, action-resource, and
  action-utility heads;
- epochs 6–10 add tactical-outcome and opponent-response heads;
- epochs 11–15 add resource-forecast and game-phase heads;
- epochs 16–20 add outcome-distribution and remaining-turn heads; and
- epochs 21–25 train the complete enabled head set jointly.

Existing policy, value, belief, strategy, game-plan, archetype, and eligible
matchup objectives continue under their established contracts. Expanded-head
losses continue during RL and scheduled expert rehearsal whenever their exact
labels are present. Checkpoints and receipts record the target and schedule
digests, architecture-present heads, gradient-enabled heads, runtime-enabled
heads, weights, train/validation losses, labeled/masked row counts, coverage,
and calibration diagnostics. Local and remote workers must agree on these
schema digests. A metadata/tensor mismatch fails closed.

The initial numerical weights and exact head dimensions are authoritative in
`config/rl_protocol.yaml#/specialist_training/expanded_strategic_heads`.
Gameplay gates do not change. Before activation, measured bootstrap/RL training
throughput may regress by no more than ten percent. The fused serving path must
publish measured rollout throughput and memory evidence, and OOM is never
accepted.

### Parallel orchestration and wall-clock throughput

The program optimizes completed training iterations per wall-clock hour.
Independent work must run concurrently across available hardware: source
download and validation, per-day featurization, corpus assembly, current-deck
guide preparation, derived CPU-pack construction, dashboard validation, and
successor-specialist readiness checks may all overlap healthy active-specialist
training and one another.

Sequential execution is permitted only for a real dependency: an input
artifact must exist before its consumer, a checksum or immutable receipt must
be committed before a bound transition, the selector update must occur at its
safe boundary, or the single-active-specialist rule requires one learner.
Controllers must not introduce global barriers for unrelated work. Parallel
jobs remain subject to the configured memory guards and may not starve,
restart, or preempt healthy production training.

#### Retrofitting expanded heads onto completed specialists

After the first cumulative core containing the expanded V6 architecture is
frozen and checksum-registered, completed V5 specialists may receive the
architecture only as new, separately identified compatibility derivatives.
The exact original passing checkpoints remain immutable and continue to be the
authoritative historical gate and Kaggle artifacts. A retrofit must never
rewrite, replace, re-freeze, or silently promote an original checkpoint.

Every added expanded head is deterministically initialized, materialized as
dormant, and runtime-disabled. Creating the derivative is architecture
migration, not training completion, gate passage, or runtime activation. The
derivative must record its source checkpoint checksum, cumulative-core
checksum, target and schedule digests, initialization seed and method, tensor
compatibility audit, and a checksum distinct from the source artifact.

A dormant retrofit may be trained later only in an explicitly scheduled
retrofit phase in which that specialist is the sole active learner. All other
completed specialists and their derivatives remain frozen and inference-only.
The retrofit uses the same exact-label masking, 25-epoch expanded-head
bootstrap schedule, rehearsal contract, and research/training separation as a
new V6 specialist. It may not affect policy actions, search, matchup routing,
public-mix inference, or holdouts until it independently passes compatibility,
regression, and gameplay validation and receives a separate activation
receipt. Until then, eligible public-mix and gate opponents continue using the
original frozen runtime package.

### Active-specialist expert corpus

Every newly built bootstrap or rehearsal corpus uses the latest 20 available
calendar days from the authoritative daily episode index, inclusive of the
newest fully validated day. After all 20 daily sources have been checksum
validated, filter their combined replay population to the current active
archetype. The 20-day source window is fixed before filtering; it must not be
shortened to only the dates on which that archetype appeared, expanded
backward to obtain a preferred sample count, or reused for a different active
archetype without rebuilding and re-pinning the filtered corpus.

The protected corpus receipt must identify all 20 calendar dates and the
archive/feature checksum for every date, including dates that contribute zero
matching games after archetype filtering. It must separately record per-day
matching-game and decision counts plus the aggregate filtered totals. The
dashboard must display all 20 source dates. A date may be shown as present only
when its daily source and derived feature shard are validated; zero matching
games is a valid present date and is not the same as a missing date.

Only one narrow historical fallback is permitted. First complete and receipt
the latest-20 window exactly as above. If, and only if, filtering those 20 days
to the active archetype produces exactly zero matching games in aggregate, the
builder may search older checksum-validated archive and feature shards for
that archetype. One or more matching games in the latest-20 window prohibits
fallback, regardless of whether the resulting decision count is considered
small.

Historical fallback never expands or replaces the latest-20 window. It must
produce a separate immutable receipt that records the latest-20 zero-match
proof, every older source date and checksum used, per-day and aggregate
matching-game and decision counts, and the active archetype. Its corpus and
monitoring status must be labeled `historical_zero_match_fallback`; neither
the artifact nor its dates may be called `latest20`. The dashboard continues
to display all 20 latest source dates, including their zero counts, and shows
fallback provenance separately. Unknown fallback search order, shard limit,
and stopping rule remain unset until explicitly authorized; they must not be
invented by an implementation.

Before bootstrap, and before a scheduled rehearsal when a newer fully
validated daily source is available, rebuild and checksum-pin the 20-day
archetype-filtered corpus. An already-running update remains bound to its exact
protected corpus; corpus refreshes take effect only at the next safe
bootstrap/rehearsal boundary and never rewrite historical receipts.

Poll the authoritative Kaggle episode index hourly so a newly published daily
dataset is incorporated without waiting for a manual handoff. Bert is the
exclusive Kaggle ingress host and must use its Wi-Fi default route. Each
missing daily archive is validated on Bert, transferred to Elmo over Bert's
Ethernet source address, checksum-validated on Elmo, and committed through an
atomic latest-20 receipt. Bert must delete its temporary replay ZIP only after
the committed Elmo receipt records the identical checksum. Inzi receives only
the small receipt; replay archives do not traverse Inzi's constrained link.
Existing checksum-valid Elmo archives are reused instead of downloaded again.
The rolling archive refresh never changes an already-running training update;
derived specialist features activate only at the safe boundary described
above.

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
checksum-verified frozen specialist and create a new versioned and checksummed
candidate. If it passes the established core acceptance checks, hot-start the
next specialist from that newly accepted core. If it fails, preserve it as
rejected diagnostic evidence and immediately hot-start the next specialist
from the latest previously accepted core. This fallback is explicit,
checksum-bound, and nonblocking; it does not weaken the failed candidate's
gate or disable the next boundary's refresh attempt. Whether the final
strongest core should replace the bases of all completed specialists is
intentionally undecided and requires a separate explicit decision; this
protocol does not perform that replacement.

The active specialist is never preempted. At each specialist handoff, choose
the next model only from specialists that do not yet have an exact frozen
checkpoint that passed both baseline gates after the iteration floor. First
prioritize specialists for which the registry has no existing specialist
model or checkpoint artifact. Within that
group, rank by descending current public-ladder meta inclusion share from the
pinned PTCGReplay snapshot at `https://ptcgreplay.netlify.app/`. PTCGReplay is
the authoritative source for all meta analysis: archetype prevalence, matchup
analysis, play-order splits, deck-list analysis, and specialist priority.
Only after that group is exhausted may the
scheduler select an unfinished specialist that already has an existing model
artifact; rank that second group by the same meta-share rule. An artifact may
establish availability without being protocol-valid or resumable: its recorded
restart policy still applies.

Refresh and pin the newest completed PTCGReplay ingest immediately before a
handoff. The pinned receipt must record the ingest ID and timestamps, date
window, match-fact count, archetype mapping, aggregation filters, and source
schema. Derive displayed values from the authenticated per-match `match_facts`
stream using the site's own side-aware aggregation semantics; do not scrape
rendered chart text or reuse the retired PTCG Ladder Meta strategies API.
Credentials published by the site configuration are runtime access material
and must never be committed to this repository. A refresh may
reorder only specialists that have not started. Equal shares use the stable
target-registry order. Missing or ambiguous source mappings remain null and
sort after verified shares within their availability group; they must not be
guessed. Public meta share is a training-priority signal only and never changes
a gate or supplies training examples. PTCGReplay does not replace expert replay
archives, training labels, official/premium research agents, frozen-specialist
registries, or established gate evidence.

A handoff may carry an explicit, recorded operator priority prefix, owner
removal, or overlap deferral. The named missing-model targets run first in
their recorded order; the remaining targets then return to the normal
availability/meta ordering. An owner-removed target is absent from selection
and completion accounting, while historical corpus, audit, and stable matchup
slot evidence remains immutable.
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

### Canonical specialist roster and Matchup Adapter V6

The logical authoritative matchup roster is
`state/matchup_adapter_roster.json#/active_expert_ids`; it is not itself the
required specialist-plan list. The live run remains on the immutable
`poke-bot-matchup-adapter-bank-v5-roster18` format until a safe receipt-backed
boundary. The staged successor format is the roster-neutral
`poke-bot-matchup-adapter-bank-v6`.

V6 always contains 64 physical adapter slots. Slots 0 through 17 preserve the
exact V5 route identities and tensor values; unused slots are entirely zero and
have no optimizer state. Adding an archetype allocates the lowest never-used
slot. Removing one marks its slot retired and disables routing, gradients,
optimizer steps, replay updates, and rehearsal updates without deleting or
reindexing it. Retired slots are never automatically recycled. Therefore an
ordinary roster edit changes registry data, not model shape, state-dict keys,
parameter count, or checkpoint format.

Raging Bolt, Gardevoir, N's Zoroark, Lopunny, and Cornerstone Ogerpon remain
retired from the logical roster. Historical V4 and V5 artifacts remain
immutable lineage. V6 activation may not replace or rewrite any passing V5
checkpoint.

The v4 `festival-lead` route was migrated by route identity to the canonical
`thwackey` route. Team Rocket's Spidops was appended as an exact zero-output
route. Retained route tensors were required to remain byte-identical by route
identity, retired rows were deleted, the appended row was required to be
exactly zero, and adapter optimizer state was discarded during the
non-prefix-compatible migration. The migrated active checkpoint, causal
router, authorization receipt, and fleet copies must share their recorded
checksums.

Head-count and parameter checks must never encode a copied total. In V6 the
physical adapter count is the registry's fixed `slot_capacity`; the active
logical count is the length of `active_expert_ids`. The parameter expectation
is derived as:

`physical slot capacity × parameters in one adapter head as instantiated by
the checkpoint's declared per-head architecture`.

Tests and monitoring must derive both factors from those authoritative
structures. A roster change does not update the physical parameter expectation.
A per-head architecture change still requires an explicit model migration.

The V6 loader accepts only the exact known V5 18-row contract. It copies those
rows and name-keyed optimizer moments byte-for-byte into slots 0 through 17,
adds no optimizer state for unused slots, and preserves the trunk, other heads,
counters, RNG state, and provenance. Positional optimizer state without a
verified parameter-name mapping fails closed. A V6-to-V5 projection is allowed
only when no active or trained V6-only slot would be lost.

PTCGReplay is authoritative for meta analysis and specialist priority. A
separately checksum-bound exact submitted-deck-to-archetype catalog from its
public deck table may also repair a demonstrably stale local classifier for
the same public Kaggle replay window. That catalog may label only the acting
seat's exact submitted 60-card multiset; the causal decisions still come from
the original checksummed replay archive. It may not supply actions, hidden
state, outcomes, holdout results, or gate evidence. Mappings use the numeric
archetype identifier plus exact source name as a consistency guard. Fuzzy
display-name matching is prohibited, and aggregate, ambiguous, or missing
mappings do not become training labels.

After the current cumulative-core boundary, the explicit unfinished priority
prefix is Dragapult/Dusknoir, Dudunsparce, Marnie's Grimmsnarl ex, Cynthia's
Garchomp ex, Team Rocket's Mewtwo ex, Thwackey, and Team Rocket's Spidops.
Team Rocket's Spidops is the mandatory successor after Thwackey. Its absence
from a smaller expert window blocks selection and requires recovery from the
full public rolling replay history; it never permits fall-through to a
lower-priority executable corpus. The June 26 through July 27 public window
must contribute at least 16,639 checksum-bound exact acting-seat Spidops games.
Hammer-Pult then returns at its actual meta priority without the
existing-artifact priority penalty. The strict post-Spidops successor prefix is
Hammer-Pult, Teal Mask Ogerpon ex, then Archaludon ex. Missing inputs for any
member of that prefix block fallback and trigger public-data recovery.
For Teal Mask, the competitive deck taxonomy is `Ogerpon Box`, while the
current PTCGReplay public identity is numeric archetype 151 with the exact name
`Teal Mask Ogerpon ex`; those identities must not be substituted for one
another. Completed ingest 17's catalog/index binds one exact public deck
fingerprint and a 1,135-game acting-seat floor across the full June 26 through
July 27 window. Exact raw replay materialization expands that floor to 2,300
unique acting-seat games, 156,692 decisions, and 10,495 causal guide rows under
32 verified daily receipts with no duplicate episode-seat keys. It includes
all four nonzero recent days from July 24 through July 27: 42, 259, 413, and
421 games respectively. Its V6 matchup identity is allocated dormant at stable
slot 18 until its causal route passes the ordinary precision and
weighted-support audit.
Dragapult/Blaziken and Dragapult/Dudunsparce are removed from the required
specialist plan and completion count; their historical corpus, router, and
audit artifacts remain evidence only, and their stable matchup slots are not
deleted or reused. Other retained unfinished specialists resume only after
Archaludon ex.

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

Research evaluations are completely separate from training. Before any
archetype-supersession rule is applied, the catalog contains 1,000 official
games and 2,000 premium games. The exact current total is derived from the
active external premium roster after supersession plus 250 games for every
registered frozen specialist:

- Official holdout: 1,000 games against 4 official agents, exactly 250 per
  agent, with 125 going first and 125 going second.
- Premium competition holdout: exactly 250 games against each active external
  premium opponent after supersession and each frozen completed specialist.
  Every opponent receives 125 games with the candidate going first and 125
  going second. Frozen specialists are labeled `S+`, carry the established
  S-tier safety-floor requirement, and remain inference-only.

The official research-control roster is permanently fixed to exactly those
four official agents. It must not grow when a specialist is frozen, and no
frozen specialist may be placed in that roster. This restriction applies only
to research controls: every eligible frozen specialist remains a required
inference-only opponent in the separate premium/S+ holdout gate.

Lucario has one explicit supersession rule. When our Lucario specialist
passes, is frozen, and is registered, all external premium-holdout opponents
whose canonical archetype is `lucario` are removed from every subsequent
premium holdout. The exact frozen Lucario specialist remains in the holdout as
an `S+` opponent. Historical results against the removed external Lucario
agents remain immutable and visible. This rule does not alter the separate
official research-control roster and never removes any non-Lucario opponent.
In the current catalog this removes five external Lucario opponents, leaving
three external premium opponents. The frozen registry currently contains eight
specialists, so the current premium holdout is 2,750 games across eleven
opponents and the official-plus-premium research total is 3,750 games. These
two current totals grow by 250 whenever another completed specialist is frozen
and registered; the three-external count remains unchanged unless another
explicit supersession decision is recorded.

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
research results, but it cannot complete or transition before its iteration-5
commit. The exact 250-game-per-opponent allocation, active external roster
after explicit supersession plus every frozen specialist, both-seat balance,
weighted-win-rate floor,
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
gate at or after the iteration-5 floor:

1. Freeze and register the exact passing checkpoint and checksum.
2. Record all reproducibility metadata and complete holdout results.
3. Mark the specialist training-complete. Kaggle submission completion is a
   separate asynchronous obligation and is not part of training completion.
4. Create one clearly labeled submission-copy record pinned to the checksum of
   that exact frozen passing checkpoint.
5. Automatically create exactly one single-use Kaggle authorization bound to
   the specialist ID, frozen checkpoint checksum, upload-bundle checksum,
   competition, and label. This standing authorization rule applies at every
   training-complete boundary; it is not a reusable or unbounded grant.
6. Submit that copy if the remaining five-submission daily quota and four-hour
   spacing permit, then record its submission ID, timestamp, and returned
   score.
7. If quota is exhausted, mark the copy `pending`, append it to the
   persistent submission queue, and immediately begin or continue the next
   unfinished specialist.
8. Add the frozen checkpoint to the eligible public-mix inference-only pool
   and to every later premium holdout as an `S+` opponent.
9. Immediately begin the next unfinished specialist.

Before an upload, the submission builder and asynchronous queue processor must
both fail closed unless the package contains: (a) the exact frozen passing
checkpoint bytes and matching specialist archetype ID, and (b) the exact
60-card representative bound by checksum to that specialist run's measurement
deck contract, and (c) that specialist's validated checksum-pinned causal
matchup tree. The queued label, specialist ID, checkpoint checksum, deck-file
checksum, canonical card-list checksum, representative-registry checksum, and
matchup-tree checksum must remain immutable. A mismatched or stale
model/deck/router package is failed and must never reach Kaggle.

If a prior attempt was rejected locally before network I/O solely because its
one-shot authorization was missing, restore that exact queue item to `pending`
and issue the same checksum-bound single-use authorization oldest-first. Never
use this recovery rule to retry an upload with an unknown network outcome.

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
