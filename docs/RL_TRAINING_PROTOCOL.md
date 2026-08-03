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
cumulative policy and hot-starts the next specialist from that accepted
fallback. Accepted Policy Generation 9 is the current accepted fallback. The
post-Thwackey Policy Generation 10 candidate is preserved as a rejected gameplay-regression
attempt and is not bootstrap-eligible. The post-Spidops Policy Generation 11 attempt is likewise
preserved as a rejected pretraining-validation attempt: it created no candidate
and ran no gameplay regression. Hammer Pult therefore started from
Accepted Policy Generation 9. Matchup Router Format 6 is a separate checkpoint-layout
version and must never be displayed or interpreted as the Training Core
Revision or Accepted Policy Generation. The
controller still attempts a new cumulative refresh from all frozen teachers
after every later completed specialist.

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
specialist-specific current-deck guide contract before bootstrap. A sparse
teacher ranks the complete legal action stage only when cited strategy
evidence and the causally observable public state support a high-confidence
preference. In the already-started legacy Slop Box lineage only, the resulting
masked, confidence-weighted guide cross-entropy may shape shared policy logits.
For every future run beginning with Archaludon ex, the guide instead selects
relevant causal rows and scales only outcome-backed strategic-head objectives;
the guide preference index is not a policy target and direct guide-to-policy
cross-entropy is forbidden. The guide is not a serving action path, may not
inspect hidden or future state, and never overrides the authoritative fused
policy.

The guide is temporary scaffolding. Its loss weight ramps from 0.01 to 0.05
during bootstrap epochs 1–5 and remains at no more than 0.05 through the exact
25-epoch bootstrap. Post-bootstrap every future training run covered by the
prospective policy follows the owner's realized-importance curve. Separate
training-ineligible guide-on/guide-off evidence may
ramp the weight through 0.15, 0.25, 0.35, and 0.50 while its realized-win lower
confidence bound remains positive. The guide then holds at its evidence-
supported plateau and decays toward zero as the policy internalizes it or its
marginal contribution becomes non-positive. Revision 43 authorizes the fleet
controller to make those adjustments without a new per-deck decision, but
revision 44 scopes that authority to future specialist training runs only,
beginning with Archaludon ex. It does not retrofit any completed, frozen, or
already-started run. Historical checkpoints, guide weights, and receipts remain
byte-for-byte unchanged. Every prospective change still requires a
checksum-bound clean-boundary receipt. The absolute auxiliary ceiling is 0.50.
The future-policy runtime modules are installed atomically only after the
predecessor training service is verified inactive and immediately before the
future specialist is registered. The install uses no service control, records
the exact source and target checksums in a
`poke_bot.future_specialist_guide_weight_policy_install/v1` receipt, and fails
closed if an active predecessor or changed staged module is observed.
For an eligible future run, the current receipt-backed weight is not merely
descriptive: it is the literal multiplier on bounded, guide-conditioned losses
whose labels remain observed causal strategic-head targets. It changes the
gradients that train those independently computed learned heads and their
shared representation; it does not create a guide-imitation target or a direct
guide-to-policy loss and remains absent from the serving-time action-selection
path.

For the active Slop Box lineage, revision 42 authorized one historical increase
from 0.05 to 0.25 after the immutable iteration-5 commit, and revision 50 later
returned the legacy guide weight to exactly 0.05 after the immutable
iteration-12 commit because the higher imitation pressure did not improve
wins. That run remains an explicit legacy exception; the revision-43 automatic
review lifecycle is not attached to it. For future runs, plateau and decay
decisions use balanced,
training-ineligible and replay-ineligible paired guide-on/guide-off
counterfactual evaluations, reported overall and by matchup. The counterfactual
checkpoint is never eligible for serving or promotion. A positive lower
confidence bound holds the plateau; two consecutive non-positive reviews move
through 0.15, 0.075, then 0.0. Training outcomes and formal gate games cannot
control this schedule. Every change requires a checksum-bound schedule receipt.
Each fleet review contains at least 1,000 exact matched guide-on/guide-off
opponent-seat-seed pairs, at least 50 pairs for every reported matchup, balanced
first/second seats, and a 90% one-sided realized-win-delta lower confidence
bound. `poke_bot/pure_rl/guide_weight_evidence.py` validates those isolation and
identity guarantees and `scripts/compile_guide_weight_schedule.py` emits the
immutable schedule receipt. Newly registered future guide specialists checksum-bind
the generic schedule, evidence, and review-request modules before launch.
After every committed five-iteration boundary in a run started under this
prospective policy,
`poke_bot/pure_rl/guide_weight_review.py` automatically emits an immutable,
non-blocking shadow-pair request bound to the exact five collection receipts,
seed checkpoint, guide contract, and iteration commit. The production learner
does not run the shadow fit inline and never changes a weight from the request
alone; only separate paired evidence plus a compiled schedule can authorize
the next clean-boundary weight.
The managed future-only shadow queue consumes that request outside the
production learner. It trains guide-on and guide-off checkpoints from the same
parent, five committed replay shards, split, batch order, optimizer, and seed;
the guide multiplier is the only changed training input. It then runs 1,000
greedy matched opponent/seat/seed pairs locally without publishing either
checkpoint to serving workers. The worker is CPU- and I/O-bounded, is
ineligible for active Teal, and writes only training-ineligible,
replay-ineligible, non-gate evidence. Its compiled schedule is passed to the
managed clean-boundary controller. Weight-hold reviews still persist their
non-positive evidence counter through the same receipt-backed restart so the
second consecutive non-positive review cannot be forgotten.
Because the 1,000-pair shadow study is deliberately non-blocking, its schedule
records the earliest eligible next iteration rather than claiming that the
study will finish during the same 30-second pause. The generic boundary
controller applies a completed schedule at the first later available
five-iteration hard pause and writes the actual application iteration into a
`poke_bot.future_specialist_guide_weight_boundary/v1` receipt.

This ramp/hold/anneal lifecycle applies to every future specialist training run
started under the prospective policy and is the owner's protected goal-path
guide vision. It may not be silently removed, replaced with a permanently
fixed weight, or driven by ordinary training outcomes. It remains a required
part of future guide contracts alongside the dedicated two-task research
workflow. It may not be backfilled into completed, frozen, or already-started
runs.

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
frozen Model Format 5 checkpoint is rewritten. During each ordinary full-model RL epoch and
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

Beginning with the Archaludon ex pre-stage, a future specialist's deck guide
is a bounded curriculum for causal strategic objectives rather than a
policy-imitation target. Every strategic head is an independently computed
view of the current causal board state and, where option-conditioned, each
current legal option. Every such view must enter the checksum-bound learned
decision-fusion action score through its own explicit bounded route:

- `computation_role: independent_head`;
- `fusion_role: fused_input`; and
- `action_influence: bounded_option_conditioned_route`.

“Independent” describes the computation before fusion; it does not permit a
head to be omitted from action choice. A state-targeted head may keep its
state-level supervised target, but its action route combines that typed output
with each legal option representation after board/state cross-attention. An
option-targeted head combines its typed per-option output with the same causal
option representation. The future fusion schema owns one attributable bounded
route per learned decision head and adds their capped aggregate to the
preserved parent fusion output; it may not first average every state head into
one option-blind context. The guide itself is the sole exception: it is
training-only curriculum metadata, not a learned runtime head, is not a
runtime input, never directly selects an action, and never replaces observed
outcomes. This future-only rule does not retrofit Slop Box or any completed,
frozen, or already-started specialist, and it does not authorize pre-fleet H10
computation.

The first required new branch is `setup_board_outcome`. It is an
option-conditioned `d_model → 512 → 9` MLP that trains only on selected,
complete setup-active or setup-bench stages. Its nine outputs reuse existing
causal targets: the six next-own-decision resource-forecast components and
terminal loss/draw/win. Unchosen candidates, incomplete games, unavailable
resource components, malformed stages, and non-setup contexts are masked.
Setup-active and setup-bench losses are reduced separately and then averaged,
so rare bench-development rows cannot disappear inside the main-phase volume.
The ordinary head weight is `0.025`.

On a high-confidence guide-labeled setup row, the guide multiplier may add
`guide_confidence × observed-target loss` for this branch. The guide's
preferred action index is deliberately not consumed: permuting that index must
leave the setup-head loss and gradients unchanged. The observed next board and
game outcome supply the learning direction. The branch may update its own MLP
and the shared option representation. Its option representation cross-attends
the current board/state, and its nine typed outputs enter a bounded, zero-safe,
versioned decision-fusion route. The zero-safe initialization preserves the
parent's exact logits and choices at migration; after training, runtime
activation requires measured finite nonzero action-logit influence and
leave-one-head-out attribution.

Future guide predicates must also implement their stated causal
preconditions. If a guide says a bench choice depends on a resource such as
Energy support, that support must be proven from the permitted current state
or the entire stage abstains. STOP and non-STOP candidates are scored under
the same preconditions. This prevents an unconditional positive Basic score
from turning every legal bench opportunity into a supposed policy error.

Archaludon bootstrap remains fail-closed until its validation receipt binds the
training mode, head-role map, curriculum map, code and target digests, and
proves selected-only masking, target parity, guide-index permutation
invariance, nonzero branch/shared-representation gradients, exact step-zero
parent action parity, nonzero fusion-route gradients after the zero-safe warm
step, bounded finite action-logit influence, causal board/option dependence,
one declared route and nonzero leave-one-head-out attribution for every
learned decision head, a bounded aggregate route delta, STOP/non-STOP coverage,
calibration, and update norms.

Slowking is the next planned specialist after Archaludon ex, not an alternate
successor that can bypass it. It is a combo/toolbox deck bound only to the
owner's exact 4 Slowking / 4 Slowpoke list with the Annihilape, Conkeldurr,
Kyurem, Latias ex, Mega Kangaskhan ex, Smoochum, Fezandipiti ex, Meowth ex,
Academy at Night, Ciphermaniac's Codebreaking, Poké Pad, Wondrous Patch,
Counter Gain, and Psychic/Telepath/Boomerang Energy engine. A generic
Slowking/Metagross list is not a valid representative, guide, corpus identity,
or terminal package.

Before Slowking can bootstrap, the checksum-bound
`state/slowking_combo_head_coverage_v2.json` map must prove learned decision
coverage for top-deck construction and consumption, copied non-Rule-Box attack
legality and choice, visible combo pieces and search/recovery, every relevant
Energy attachment/return/acceleration route, replacement-attacker Bench
continuity, disruption response, prize mapping, and remaining-turn/outcome
timing. Existing action, resource, tactical, opponent-response, prize-race,
outcome, remaining-turn, and setup heads cover parts of that contract. They do
not by themselves provide the explicit causal targets for known-top provenance
or copied-attack legality. The implemented generic `combo_state` head provides
those targets through a `d_model → 192 → 32` option branch and its own
revision-56 bounded option-conditioned fusion route.

The schema-8 full-history corpus now proves 311 exact-deck games, 19,251
combo-labeled decisions, 887 top-deck labels, 4,948 seek-source labels, and
nonzero copied-attack, visible-piece, Energy-route, and Bench-continuity
coverage. Its imported resident CPU pack contains the combo tensors in a
280-game train / 31-game validation split. Implementation receipts prove
finite nonzero head/route gradients and updates, step-zero parent parity,
causal suffix invariance, legal-option dependence, bounded aggregate residuals,
and leave-one-head-out logit attribution. Slowking remains fail-closed behind
Archaludon and the trained-candidate calibration, action attribution, ablation,
runtime-resource, package, and terminal-preflight receipts. The guide remains
the sole no-route exception and may only weight observed-target learning; it
is never an imitation policy or serving input.

The ordinary two-million-parameter goal is a soft target for Slowking. Its
specialist-scoped exception may accept a measured candidate above two million
only when the receipt accounts for every added module and demonstrates useful
causal gradients and ablation gain. The initial hard ceiling remains the
existing 3.5 million parameters; no global environment default changes. The
currently proposed `combo_state` MLP contributes 24,800 parameters at d96 and
its dedicated route contributes 2,081, for 26,881 explicitly named added
parameters. The complete pre-stage candidate has been instantiated and
measured at 1,910,963 parameters, below the ordinary soft target. Exceeding
3.5 million requires a
separate owner decision plus memory, latency, package-size, gradient-use,
ablation, and final-submission compatibility receipts.

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

Completed-result compaction is part of wall-clock throughput. Beginning with
Marnie iteration 5, every self-play, public-mix, promotion, research-control,
and formal-holdout wave must enable
`POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN=1`. Once every local and remote
producer has finished and every result/done record is durably present in the
bounded RAM/disk queue, the phase-owned local simulator pool is released before
the serialized consumer compacts the remaining rows. The same lifecycle is
required for scheduled and legacy additive dispatch. This preserves every
exact row and retry/checksum audit while preventing exhausted simulator model
copies from holding the trainer near its memory-high boundary during ingest.
The Marnie iteration-4 tail near `20 s/game` and roughly eight hours is retained
as rejected historical evidence; it is not an acceptable self-play or holdout
rate and may not recur as the next iteration's operating mode. The iteration-5
activation is bound by
`/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latest20-runtime-activation-r109.json`
(`sha256:fc23eed0dabb0e42e80eadfad74f6ec6975dd68e4f3de63eda9f3accd6bcadf6`).
Its first live self-play wave released all 96 phase-owned local children before
the compaction tail; host available memory rose from about 12.4 GiB to 37.2
GiB before public mix started. The following 7,168-game public-mix wave also
released all 96 exhausted local children before its compaction tail and sealed
collection receipt
`sha256:1f7786a721df6ea22a6e6b467b5e4681c2dbbb42ababb4f4b8df3c9464c471dd`
before expert rehearsal began. Promotion, research-control, and formal-holdout
waves remain independently observable under the same mandatory lifecycle.

Future scheduled additive waves beginning with Marnie iteration 6 use a
20%-remaining tail work-stealing boundary. At that point the controller returns
endpoint-owned reservations above one execution wave to the shared exact-job
pool, remote emitters claim one game at a time, and Blackwell's exact 96-worker
pool remains eligible until global completion. This prevents a slow remote-only
tail after local completion. It does not batch network results: each completed
game still streams independently into the bounded result queue. Job identity,
seed, seat, replay-row, retry, checksum, and producer-drain audits remain exact.

Beginning at the immutable Marnie iteration-5→6 boundary, the shared
public-mix/formal-holdout roster uses architecture-aware tier weights. Every
eligible non-active H10 specialist is `S`/`2.0`; every other frozen specialist
and every remaining public opponent is `A`/`1.0`. This changes adaptive
practice allocation and the formal skill-weighted aggregate only. It does not
change the exact 17 opponent identities, their content checksums, 250 games per
opponent, 125/125 seats, the 4,250-game audit, research controls, or the
independent 80% skill-weighted and 50% confidence-lower terminal guards.

#### Retrofitting expanded heads onto completed specialists

After the first cumulative core containing the expanded Model Format 6 architecture is
frozen and checksum-registered, completed Model Format 5 specialists may receive the
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
new Model Format 6 specialist. It may not affect policy actions, search, matchup routing,
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

Elmo's production memory watchdog uses uniquely charged cgroup-v2
`memory.current` against the 30 GiB inner limit. On a host without cgroup-v2 it
falls back to process-tree PSS, and only then to summed process-tree RSS.
Summed RSS is retained as diagnostic telemetry because spawned workers share
model and library pages and summing their RSS double-counts those pages. This
measurement correction does not change the 30 GiB inner limit, 24 GiB
host-available floor, 64 GiB/no-swap outer cgroup, worker count, child
recycling, or checkpoint/capacity guards.
The activated guard is continuity-verified by
`state/elmo_memory_guard_collection_continuity_r74.json`: the complete
Archaludon iteration-3 4,000-game heldout dispatch drained with 1,632 remote
games, zero Elmo restarts, zero failed jobs, and zero cgroup memory events.
The rolling archive refresh never changes an already-running training update;
derived specialist features activate only at the safe boundary described
above.

Blackwell's long-lived trainer must return freed device-resident training
blocks to CUDA before rebuilding its inference leaves. After deleting the
replay dataset and collecting the host heap, it synchronizes the training
device, records reserved bytes, empties the CUDA caching allocator, records the
post-release value, and only then restores the 4/12 leaf farm. A leaf OOM is a
failed game, never evaluation evidence; an incomplete exact-row audit remains
uncommitted and is recovered from the immutable collection and candidate.
`state/blackwell_cuda_cache_boundary_repair_r75.json` binds the active repair
and its clean 4,000-game formal plus 1,000-game diagnostic replay. The first
ordinary post-repair training boundary is independently sealed by
`state/blackwell_cuda_cache_boundary_continuity_r75.json`: iteration 4 released
CUDA-reserved memory from 45,116,030,976 bytes to 104,857,600 bytes before
rebuilding the unchanged 4/12 leaf farm, completing promotion, and dispatching
the next 4,000-game formal evaluation with 64 local and 52 remote slots.
`state/blackwell_cuda_cache_evaluation_continuity_r75.json` seals the completed
continuity result: 4,000/4,000 formal rows passed the exact audit at 68.51%
skill-weighted win rate, the disjoint 1,000/1,000 research-control rows passed
at 59.9%, iteration 4 committed, and the required iteration-5 curriculum began
without a CUDA OOM, failed leaf, traceback, or service restart.

Those 64-local evaluations are immutable historical evidence, not the current
worker default. Under the owner-pinned Blackwell profile, every current and
future collection, formal heldout, and evaluation pool inherits exactly 96
local simulator workers and 96 local games in flight. The runtime must also
pin `PURE_RL_HELDOUT_LOCAL_WORKERS=96`; a missing heldout override may not
silently fall back to 64. The worker floor, target, ceiling, rebalance bounds,
and live-pool maximum remain 96, with memory guards failing closed rather than
substituting a smaller pool.

The current Archaludon run intentionally uses the already imported protected
dataset-schema-6 corpus authorized by goal revision 69. Dataset schema 7 added
only exact setup `SelectContext` and demonstrated STOP metadata. The
revision-76 compatibility reader therefore preserves every earlier causal
target and maps only those unavailable setup fields to UNKNOWN/false; UNKNOWN
context masks the setup objective rather than creating a zero target. The
runtime pack must still be rebuilt and verified under its exact packing-schema,
split-seed, manifest, vocabulary, and checksum key; an older pack may not be
renamed or relabeled. Receipt
`state/archaludon_expert_rehearsal_schema6_compat_repair_r76.json` proves the
exact key `7c5091c9d2d4fe6d1ae6ddd39373082014b8e9c3be6da01b68037218f5ba253d`
loaded as a cache hit, the five-epoch rehearsal committed, and iteration 5
resumed from its unchanged 8,192-game collection without recollection.

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
The current owner-removed targets are plain Dragapult, Dragapult/Blaziken,
Dragapult/Dudunsparce, Crustle, and Walrein. Plain Dragapult's completed
corpora, route, guide, and audits remain immutable non-planning evidence.
Slowking replaces that required-fleet slot after Archaludon ex and is the last
remaining specialist. Its successful freeze and registration immediately
trigger the separately versioned final-format Alakazam refresh described
below; no final-format model computation is allowed before that boundary. Its exact
owner-shown 60-card list, expert guide, generic guide registration, canonical
representative, and blocked pre-stage package are discoverable under
`config/deck_guides/slowking.yaml`,
`config/deck_guides/slowking-representative.v1.json`,
`docs/deck_guides/slowking-expert-brief.txt`,
`state/slowking_combo_head_coverage_v2.json`, and
`poke_bot/slowking_heuristics.py`. This preparation does not make Slowking
selectable or trainable: Archaludon ex must complete first, and the trained
Slowking candidate still requires calibration, ablation, action-attribution,
runtime-resource, package, terminal-preflight, and final validation receipts.
Crustle is not a required specialist and may not be selected, bootstrapped,
trained, gated, frozen, submitted, or counted toward completion. Its existing
Matchup Router Format 6 identity remains active at stable slot 0 and may not be
deleted, disabled, reindexed, or reused. The canonical inference-only public
practice/gate opponent is `pilkwang-meta-20260708`, archetype `crustle`,
label `Crustle / Great Tusk`, source
`pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-08-july`, content digest
`sha256:7120bc67415e06c1cf69d64574f1a41545fd4c2fd084a029d77c5e43a357957f`.
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

### Canonical specialist roster and Router Format 6

The logical authoritative matchup roster is
`state/matchup_adapter_roster.json#/active_expert_ids`; it is not itself the
required specialist-plan list. The live run remains on the immutable
`poke-bot-matchup-adapter-bank-v5-roster18` format until a safe receipt-backed
boundary. The staged successor format is the roster-neutral
`poke-bot-matchup-adapter-bank-v6`.

Router Format 6 always contains 64 physical adapter slots. Slots 0 through 17 preserve the
exact Router Format 5 route identities and tensor values; unused slots are entirely zero and
have no optimizer state. Adding an archetype allocates the lowest never-used
slot. Removing one marks its slot retired and disables routing, gradients,
optimizer steps, replay updates, and rehearsal updates without deleting or
reindexing it. Retired slots are never automatically recycled. Therefore an
ordinary roster edit changes registry data, not model shape, state-dict keys,
parameter count, or checkpoint format.

Raging Bolt, Gardevoir, N's Zoroark, Lopunny, and Cornerstone Ogerpon remain
retired from the logical roster. Historical Router Format 4 and Router Format 5 artifacts remain
immutable lineage. Router Format 6 activation may not replace or rewrite any passing Model Format 5
checkpoint.

The v4 `festival-lead` route was migrated by route identity to the canonical
`thwackey` route. Team Rocket's Spidops was appended as an exact zero-output
route. Retained route tensors were required to remain byte-identical by route
identity, retired rows were deleted, the appended row was required to be
exactly zero, and adapter optimizer state was discarded during the
non-prefix-compatible migration. The migrated active checkpoint, causal
router, authorization receipt, and fleet copies must share their recorded
checksums.

Head-count and parameter checks must never encode a copied total. In Router Format 6 the
physical adapter count is the registry's fixed `slot_capacity`; the active
logical count is the length of `active_expert_ids`. The parameter expectation
is derived as:

`physical slot capacity × parameters in one adapter head as instantiated by
the checkpoint's declared per-head architecture`.

Tests and monitoring must derive both factors from those authoritative
structures. A roster change does not update the physical parameter expectation.
A per-head architecture change still requires an explicit model migration.

The Router Format 6 loader accepts only the exact known Router Format 5 18-row contract. It copies those
rows and name-keyed optimizer moments byte-for-byte into slots 0 through 17,
adds no optimizer state for unused slots, and preserves the trunk, other heads,
counters, RNG state, and provenance. Positional optimizer state without a
verified parameter-name mapping fails closed. A Router Format 6-to-5 projection is allowed
only when no active or trained Router Format 6-only slot would be lost.

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
July 27 window. The clean exact-archetype-151 materialization contains 1,135
unique acting-seat games, 76,226 decisions, and 6,814 causal guide rows under
32 verified daily receipts with no duplicate episode-seat keys. The earlier
2,300-game artifact is bound to Mega Kangaskhan source identity and remains
quarantined and non-promotable. Teal's Router Format 6 matchup identity remains allocated at
stable slot 18. The combined inactive Router Candidate 44 passed all 19 routes under
the unchanged precision, weighted-support, and zero-state-bypass audit and is
registered for boundary-only activation; it does not change the current
production router.
Plain Dragapult, Dragapult/Blaziken, Dragapult/Dudunsparce, and Crustle are
removed from the required specialist plan and completion count; their
historical corpus, router, guide, and audit artifacts remain evidence only,
and their stable matchup slots are not deleted or reused. After Teal, the
remaining unfinished specialist order is exactly Archaludon ex then Slowking.
Slowking preparation may run in parallel, but it cannot receive selection or
launch authority before Archaludon ex completes and its own fail-closed
receipts pass. Crustle remains a public opponent and active matchup route only;
it is not a third specialist step.

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
three external premium opponents. The frozen registry currently contains
twelve specialists, so the current premium holdout is 3,750 games across
fifteen opponents and the official-plus-premium research total is 4,750 games.
These two current totals grow by 250 whenever another completed specialist is
frozen and registered; the three-external count remains unchanged unless
another explicit supersession decision is recorded.

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
available. Reconcile Kaggle and local queue rows into logical submissions,
deduplicating the same upload by submission ID or checkpoint-bound label, then
anchor the unchanged four-hour spacing window to the second-most-recent
logical submission—not the newest row. With fewer than two prior logical
submissions there is no spacing anchor. The processor records the next eligible
upload time and leaves an otherwise-ready copy pending until that time. This
spacing wait is not a failure and must not interrupt active specialist
training. Queue work runs without interrupting active specialist training.
After Kaggle reports that the daily limit has been reached, do not repeatedly
retry during the exhausted quota window. Every delayed copy remains pinned to
its original frozen passing checkpoint: delay must never cause retraining,
modification, replacement, or
re-freezing of the specialist.

The revision-33 Hops exception is a single-use operational request for exactly
one additional `hops-trevenant` upload from the immutable owner-accepted
iteration-10 checkpoint and exact representative. Only that upload uses
`second_if_allowed`; it does not alter Hops' default behavior, replace either
historical Hops submission, or change any future specialist profile.

No completed specialist may be updated until every required specialist is
gate-passing frozen.

## 5a. Ordered post-fleet specialist refresh

After all 15 current required specialists are training-complete, frozen, and
registered, run a dedicated refresh phase before population training. The
strict order is Alakazam first and Marnie's Grimmsnarl ex second. These are new
separately versioned model generations under the same logical archetypes; they
are not new required-roster rows, do not change the required count of 15, and
must never overwrite, replace, re-freeze, or relabel either original passing
checkpoint or its historical gate and Kaggle evidence.

Slowking is the terminal required specialist, so its successful freeze and
registration immediately starts the Alakazam refresh. Alakazam is the first
model built in the final-submission format specified by
`docs/FINAL_MODEL_CAPACITY_AND_DECISION_REPLAY_PLAN.md`; it requires that
format's checksum-bound compatibility, validation, packaging, and causal-route
receipts and may not silently fall back to the legacy format. Prefer a
function-preserving migration from immutable existing Alakazam checkpoint
`sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea`
when exact key/shape coverage permits it. Genuinely new structures use
zero-safe initialization and require step-zero parity and causality receipts.
The original checkpoint remains byte-for-byte immutable. If direct migration
fails closed, preserve its failure receipt and discard the partial child.
Then build an ordinary new same-archetype Alakazam refresh from the then-latest
checksum-accepted core under the revision-36 fallback contract, and expand only
that completed Alakazam derivative to final format. The generic core is never
the direct tensor parent of final-format expansion, and no partial
old-Alakazam/core overlay is allowed.

“Immediately starts” means the controller enters the dedicated Alakazam bridge
lane; it does not authorize model computation by itself. Before any
final-format Alakazam model work, both
`required_specialist_fleet_complete_for_final_alakazam_v1` and
`capacity_research_resource_lease_v1` must exist and pass. Those receipts
authorize only the final-format Alakazam refresh. The broader
multi-archetype capacity program remains blocked until
`post_refresh_sequence_complete_for_capacity_v2` proves that both the
final-format Alakazam refresh and the following Marnie's Grimmsnarl ex refresh
are complete. No applicable receipt means no model work.

This Alakazam refresh trains on a deterministic exact even 50/50
candidate-first and candidate-second seat split. A checksum-bound
`poke_bot.alakazam_refresh_seat_split/v1` receipt must prove the final
first-seat and second-seat counts are equal before the result can pass. It
packages one `first_if_allowed` preference. It may not use the
`second_focus_1_to_7` curriculum, an always-second arm, or a second-preferring
Alakazam refresh copy. Marnie's Grimmsnarl ex remains the second refresh unless
a later owner decision supersedes it.

The active final-format refresh uses 16,384 games per iteration and one learner
epoch. After iteration 0 completed its ordinary 6,144-decision epoch but OOMed
on the first adapter-only batch, recovery reuses the exact completed corpus and
replay cache without recollection. After dense 4,096-decision optimizer batches
also poisoned the CUDA context, iteration 0 recovers at 2,048 decisions and
future warmup and ordinary learner batches initially used the measured middle
cap of 3,072 decisions. Revision 102 reduced those optimizer paths to 2,048.
At uncommitted iteration 15, the Fusion-v3 learner exhausted GPU 1 even at
2,048, with allocator failures followed by a poisoned CUDA context. Revision
105 therefore resumes the preserved iteration-15 collection and completed
rehearsal at exactly 1,536 ordinary and warmup decisions; it must not recollect
or regenerate rehearsal work. This recovery activated through design migration
`migration_0023.json`: the managed learner reused the sealed collection,
rehearsal, and seat receipts and advanced through optimizer batch 36 without an
allocator failure, beyond the prior batch-24 crash. The immutable activation
receipt is
`state/final_format_alakazam_iteration15_allocator_recovery_activation_r105.json`.
After the repaired 2,048-decision iteration-0 adapter phase
proved 27.8 GiB of free headroom, future adapter fitting also uses 3,072,
releases unused CUDA cache before fitting, and recursively splits
an OOMing multi-game batch without dropping a sequence. A single-sequence OOM
still fails closed. The adapter bank alone receives gradients in that phase;
all base tensors remain frozen and every learned head remains present in the
ordinary fused-policy epoch.

Parent/candidate policy agreement is inference-only and does not inherit the
optimizer cap. It uses the previously completed 6,144-decision inference cap
and a policy-only forward that retains deterministic argmaxes while bypassing
value, guide, AWR, auxiliary-loss, and diagnostic calculations. Exact argmax
parity with the reference loss path is required before activation.

Beginning at the first safe process boundary after iteration 0, the immutable
final-format Alakazam terminal gate requires a complete 4,250-game premium
evaluation with at least `0.75` skill-weighted win rate and a separate `0.60`
skill-weighted confidence lower bound. The disjoint official-control floor
remains `0.60`, and the actual-simulation Kaggle-rating 90% lower bound remains
an independent `1000`. An evaluation already started under the earlier 65%
contract is never reinterpreted in place.

Kaggle milestone snapshots are nonterminal observations. After durable
zero-indexed commits `4`, `9`, `14`, `19`, and every later `5n+4` commit through
`184`, an idempotent managed watcher packages one exact `first_if_allowed`
copy and appends it to the asynchronous one-shot queue. Each row binds the
commit, checkpoint, exact 60-card Alakazam representative, matchup tree, and
bundle digests. Packaging, quota waits, spacing waits, and uploads never
freeze, select, stop, pause, or block the continuing trainer.

The final-format Marnie's Grimmsnarl ex refresh uses the same nonterminal
watcher and also includes an explicit zero-indexed iteration-`0` snapshot.
Its cadence is therefore `0`, `4`, `9`, `14`, and every later `5n+4` durable
commit available before refresh completion. Every Marnie milestone is one
exact `first_if_allowed` copy bound to the Marnie commit, checkpoint, 60-card
representative, matchup tree, and bundle. These snapshots neither replace nor
consume the separately required terminal completion submission.

Resolve inputs independently at the start of each refresh. Hot-start from the
newest checksum-accepted cumulative core available at that boundary, not from
a core generation named in planning prose. Attempt the normal cumulative-core
refresh after the preceding frozen model; if its candidate is rejected,
preserve the rejection immutably and use the newest accepted fallback. After
the new Alakazam version passes and freezes, perform that normal core boundary
before resolving the base for the Grimmsnarl refresh.

The Grimmsnarl refresh resolves current weights at that boundary but not a
smaller shape: its required final shape is H10-I with seven spatial, three
temporal, and seven option-decoder layers, FF width 2,496, 512-wide strategic
head residuals, and 19 learned heads with 19 distinct action routes. It uses
typed-output-centered Fusion v3 and bounded learned route reliabilities. Its
guide is training-only `strategic_directional_v2`; guide preference ranks the
declared causal strategic routes and never supervises final policy logits.

Also checksum-bind the then-current complete canonical specialist-training
contract at each bootstrap. At minimum it must retain the exact 25-epoch
bootstrap, current latest-20 specialist corpus and current-deck guide,
causally masked learned decision fusion with every then-canonical available
head, current matchup architecture, 8,192 new games per RL iteration, the
5-RL-epoch/5-expert-rehearsal-epoch cadence, terminal preflight, and unchanged
gameplay gates. Today's Accepted Policy Generation 9 number, current schema
digests, or an old runtime package must not be pinned as the future contract.

This is full current-structure training, not the old dormant expanded-head
compatibility materialization. In particular, the Model Format 5-bound dormant Alakazam
derivative is evidence that the old checkpoint could be migrated; it is not
the requested new Alakazam version and cannot satisfy this phase. Each new
version requires its own current-gate pass, immutable freeze, version-preserving
registration, and checksum-bound
`poke_bot.post_fleet_specialist_refresh_completion/v1` receipt. The historical
Alakazam waiver and both original checkpoint rows remain evidence only.

The phase is staged as future work only. Neither refresh enters the current
selector, one-ahead pre-stage, runtime registry, service set, training queue, or
gradient scope while the required fleet is still training. At the
fleet-complete boundary, the ordinary cycle controller and direct population
preparation both fail closed until the ordered pair of refresh receipts is
complete.

## 6. Population phase

After every required specialist is gate-passing frozen and both ordered
post-fleet specialist refreshes have independently passed, frozen, and
registered:

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
