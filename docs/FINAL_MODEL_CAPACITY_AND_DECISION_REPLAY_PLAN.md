# Post-Fleet Final Model Capacity, Decision Replay, and Submission Plan

Status: authorized static implementation preparation; all model computation blocked

Plan type: receipt-gated static implementation-preparation companion

Revised: 2026-07-30

Runtime authority: none

Model-computation authority: none

Implementation authority: static schemas, receipt templates, tests, package
manifests, and release-check specifications only

Canonical planning basis: `GOAL.md` Revision 64, including owner decisions
55–59 and 61–64.

This document prepares a later receipt-gated research and final-submission
phase. It authorizes only non-computing static preparation. It does not
authorize model instantiation, checkpoint conversion, replay scanning,
benchmarking, training, evaluation, service or selector changes, a runnable
package, a Kaggle submission, or production resource reservation. Static code
or schema changes created under this authority must remain unreachable from
the active runtime and must not load a checkpoint or materialize data.

It is separate from:

- `docs/COMPLETE_FLEET_TOP_FINISH_PLAN.md`, which owns old-system fleet
  completion, practice-partner use, finalist-deck selection, and release;
- `docs/TOP_100_TURN_ORDER_OPTIMIZATION_PLAN.md`, which owns the always-second
  hypothesis and its exact `1:7` training curriculum; and
- `GOAL.md` and its typed canonical sources, which alone can authorize a later
  production transition.

The plan may accept an optional going-second training arm, but that arm is off
by default and remains a clean layer after the base architecture comparison.
The Revision-64 Alakazam bridge is the explicit exception: its training seats
are exactly even and its package preference is `first_if_allowed`. This
planning authority still does not authorize construction or submission of that
package before its release gates.

## 0. Executive recommendation

Yes, this can be staged without disrupting production, provided the following
rule is literal:

> No capacity-expansion computation begins before the exact 15-specialist fleet
> is immutably complete through Slowking. At that boundary, and only with a
> separate resource-isolation and release receipt, the required up-to-date
> Alakazam refresh becomes the first final-submission-format model. Broader
> multi-archetype capacity research still waits for Alakazam, the following
> Marnie's Grimmsnarl ex refresh, and every intervening cumulative disposition
> to be immutably resolved.

Before that boundary, only the static preparation authorized above is
permitted. The required specialist fleet is exactly 15. After the active Teal
Mask Ogerpon ex lineage, the only remaining specialist sequence is
Archaludon ex, then Slowking. Archaludon remains the hard next readiness and
training gate; Slowking cannot start first. After Slowking freezes and
registers, the separately gated post-fleet sequence begins immediately with a
new Alakazam refresh in this plan's validated final-submission format, followed
by the Marnie's Grimmsnarl ex refresh.

The Alakazam final-format refresh preferentially hot-starts from the immutable
existing Alakazam passing checkpoint,
`sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea`.
Direct migration is allowed only when a
checksum-bound receipt proves complete serialized-key and shape coverage,
causal feature/option compatibility, zero-safe initialization for every
genuinely new head and route, step-zero action/logit parity within the declared
bound, and exact role/route binding. The historical Alakazam checkpoint is
never rewritten. If direct migration fails, preserve that failure and create
an ordinary same-archetype Alakazam refresh from the then-newest
checksum-accepted core before expanding that new derivative into the final
format. Never partially overlay tensors from the old Alakazam and a generic
core.

Crustle is not a trainable specialist and does not count toward the 15. Its
stable matchup-router slot, causal route, guide, corpus, classifier, and audit
artifacts remain preserved historical/non-planning evidence; the slot is not
deleted, disabled, reindexed, or reused. The existing inference-only public
opponent remains its practice/gate implementation:
`pilkwang-meta-20260708`, archetype `crustle` / “Crustle / Great Tusk,” digest
`sha256:7120bc67415e06c1cf69d64574f1a41545fd4c2fd084a029d77c5e43a357957f`,
from `pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-08-july`.
It receives no gradients and is never frozen, submitted, or counted as a
specialist.

The main architecture bet remains **H10-I**:

```text
d_model                         96
attention heads                  8
spatial / temporal / option    7 / 3 / 7
FF width                       2496
history context                 320
fusion width                     16
learned decision routes          late-bound from final role/route inventory
minimum successor inventory      18 (historical 17 + setup_board_outcome)
possible Slowking combo route    +1 only if its coverage gate requires it
current-deck guide routes          0 (training-only metadata)
```

Every learned decision head, including `setup_board_outcome` and any validated
Slowking combo-state head, retains an independently computed causal view and a
distinct bounded action route. Each route combines that typed head output with
the current legal-option representation after the option has cross-attended
the causal board/state. The current-deck guide is the sole no-route exception:
it remains training-only curriculum metadata and never becomes a runtime
tensor or action input.

Slowking's ordinary two-million-parameter specialist target is a soft target,
not a launch ceiling, when checksum-bound combo coverage and ablation evidence
justifies extra typed capacity. Its initial specialist-scoped hard ceiling
remains 3.5 million. This exception may affect the late-bound parent inventory,
but it does not authorize pre-fleet H10 computation or create a global
parameter-limit override.

For the historical roster18/V5-sized, 17-input, 11-branch parent, the old
H10-I arithmetic produced `10,080,599` learned parameters and added
`8,396,722` parameters. Those values are historical references only. They are
not the final implementation count because the post-fleet learned-head and
distinct-route inventory is late-bound. The post-fleet migration receipt must
instantiate and publish the actual module-by-module count.

Include one learned single-pass search candidate, **H10-I+S1**, but do not
bundle it into the initial H10-I result. S1 adds a masked,
option-conditioned continuation-advantage prediction:

```text
96 → 512 → 1
```

The head MLP itself has `50,177` parameters. The old scalar-column design added
48 more for a historical `50,225` total, but that integration is superseded:
S1 must use the next distinct option-conditioned bounded route under the final
route schema. Its route size and the H10-I+S1 total are therefore late-bound.
S1 remains absent from the finalist unless it beats matched H10-I gameplay.

The owner may provide exactly **two or three finalist archetypes**. All are
screened from their own final old-system parents. With three inputs, all three
may enter the first screen, at most two continue through the expensive middle
stage, and at most one receives the first S1 and going-second experiments.

The plan also contains a post-fleet ladder emulator. It combines:

- the repository's local Elo/WR/strength-of-schedule formula;
- immutable local matchup matrices for the two or three candidates;
- our timestamped Kaggle submission scores and rank observations;
- public leaderboard snapshots when available; and
- the competition's published Gaussian skill-rating behavior.

The emulator outputs a distribution over likely Kaggle skill and rank. It is a
calibrated selection proxy, not an exact reconstruction of Kaggle's private
rating implementation and never a training signal.

## 1. Hard scope

### In scope after the release barrier

- One mandatory Alakazam final-format refresh immediately after Slowking,
  before the broader two/three-archetype screen.
- Two or three owner-selected archetypes.
- One exact old-system parent per archetype.
- Function-preserving H10-I expansion.
- Preservation and verification of one distinct option-conditioned bounded
  action route for every learned decision head.
- A matched continued-current-size control.
- Conditional H10-I+S1.
- Additional ordinary fleet-curriculum epochs.
- An optional going-second derivative.
- Local ladder and Kaggle ladder-proxy reconstruction.
- Offline decision replay and later expert decision review.

### Out of scope

- Expanding every specialist.
- Training a shared core from scratch.
- Averaging specialist checkpoints into the student.
- Deck-conditioned layers or deck routers.
- A new MoE system.
- A second matchup-adapter family.
- MCTS, UCB, tree expansion, simulator rollouts at inference, or repeated model
  calls per decision.
- Replacing the current optimizer, RL loop, cadence, rehearsal system, or
  ordinary policy/value losses.
- Any pre-fleet model computation.
- Any automatic selector, registry, service, submission, or package handoff.
- Using Kaggle scores, ranks, or games as gradient-bearing data.
- Treating guide metadata as a learned serving-time tensor or action route.
- Keeping any learned decision head as a diagnostic-only shadow branch.

The existing matchup adapters remain the only opponent-conditional
mixture-of-experts-like mechanism.

## 2. Owner input contract

The Revision-64 Alakazam bridge is a mandatory single-archetype exception to
the two/three-archetype input contract below. Its archetype is fixed to
`alakazam`; it may not be replaced during that refresh. The broader finalist
screen still requires the owner's later two- or three-archetype manifest.

At the post-fleet start, the owner supplies a manifest with exactly two or
three unique canonical archetype IDs:

```yaml
schema: poke_bot.capacity_research_input/v1

archetypes:
  - archetype_id: first-candidate
    turn_order_training: canonical
  - archetype_id: second-candidate
    turn_order_training: canonical
  # Optional third row:
  - archetype_id: third-candidate
    turn_order_training: second_focus_1_to_7

capacity_profile: H10-I/v1
search_mode: conditional_S1_v2
ladder_proxy: enabled
proxy_selection_view: balanced_50_50
```

Rules:

1. There must be exactly two or three archetypes.
2. Every archetype must already have an immutable, registered old-system
   specialist checkpoint and exact deck identity.
3. The input does not select a checkpoint. Parent resolution occurs later
   through a locked evidence gate.
4. `turn_order_training` defaults to `canonical`.
5. `second_focus_1_to_7` invokes the separate turn-order contract only after
   the base H10-I architecture result is frozen.
6. A going-second training choice does not automatically create an
   always-second package.
7. S1 is conditional for each archetype and is tested first only on the
   leading H10-I candidate.
8. Replacing an archetype after parent lock creates a new experiment
   generation; it does not silently reuse results.
9. The default ladder-proxy selection view remains balanced 50/50 even when a
   second-focused training derivative exists. A later explicit turn-order
   decision may add another projection without rewriting the balanced result.

### Two-archetype allocation

- Both enter parent lock, continued-control, H10-I migration, and Block A.
- Both may continue through Block B if they pass.
- The stronger one receives the first S1 and optional going-second branch.
- The second may reach confirmation only when it remains statistically close
  enough to be a credible final-slot hedge.

### Three-archetype allocation

- All three enter parent lock, continued-control, H10-I migration, and Block A.
- At most two continue through Block B.
- At most one receives the first S1 and optional going-second branch.
- The third remains an immutable screened result unless the first two fail.

This keeps the input flexible without multiplying every expensive experiment
by three.

## 3. Authoritative completion barriers

There are now two distinct barriers. The first authorizes only the mandatory
Alakazam final-format refresh. The second authorizes the broader
multi-archetype capacity program.

Before any Alakazam final-format model computation, require an immutable
release receipt conceptually named:

```text
required_specialist_fleet_complete_for_final_alakazam_v1
```

It must bind:

- the authoritative `GOAL.md` revision;
- the exact 15-specialist required-roster digest and completed count;
- every required frozen specialist identity and checkpoint digest;
- completion of the remaining specialist order as Archaludon ex, then
  Slowking, with no third specialist fallback;
- Slowking's immutable completion, frozen registration, exact combo-head role
  map, and one-route-per-learned-head receipt;
- terminal accepted or rejected disposition for every cumulative-core attempt
  required through the Slowking boundary;
- the newest checksum-accepted cumulative-core version and digest;
- the then-current model, feature, option, temporal, strategic-head,
  target, guide, fusion, and matchup-adapter schema digests;
- frozen and runtime registry digests;
- a stable selector snapshot;
- a frozen practice-opponent snapshot, including the exact inference-only
  Crustle binding and preserved distinct Crustle matchup route;
- no active or pending old-system specialist, refresh, or cumulative-core
  transition through Slowking;
- the checksum-bound final learned-head role inventory, including
  `setup_board_outcome` and any admitted Slowking combo-state head;
- the matching one-distinct-route-per-learned-head inventory and aggregate
  residual bound;
- `guide_runtime_route_count: 0`;
- the immutable historical Alakazam checkpoint identity and model-config
  digest proposed for the preferred direct migration;
- the exact newest accepted-core fallback identity;
- `authorization_scope: final_format_alakazam_refresh_only`;
- `runtime_authority: none`;
- `selector_eligible: false`; and
- `kaggle_eligible: false`.

A rejected cumulative-core candidate counts as terminally resolved. The receipt
selects the newest checksum-accepted fallback and never relabels a rejection as
accepted.

Before the later two/three-archetype capacity screen, require a second
immutable receipt:

```text
post_refresh_sequence_complete_for_capacity_v2
```

It must additionally bind:

- the passed, frozen, separately registered final-format Alakazam refresh;
- preservation of the historical Alakazam checkpoint;
- the direct-migration receipt or the preserved failure plus exact
  same-archetype fallback-chain receipts;
- the completed Marnie's Grimmsnarl ex refresh;
- every intervening cumulative-core disposition;
- the final post-refresh role/route and package-format inventories; and
- no active or pending specialist, refresh, or cumulative-core transition.

Pending asynchronous Kaggle copies do not need to block the receipt, but the
research lane cannot read-modify-write the submission queue.

If a newer accepted core or refresh appears after either applicable receipt,
that receipt is invalidated. Parent selection must be repeated. A running
final-format or H10 experiment never silently changes ancestry.

## 4. What may happen before the Slowking completion boundary

### Allowed now

- This plan.
- Architecture formulas.
- Input, receipt, and namespace specifications.
- Static role/route manifests and schema definitions.
- Static migration, replay-fidelity, release, package-manifest, and submission
  preflight test cases.
- Non-runnable package manifests and receipt templates.
- Search-target definition on paper.
- Evaluation and rollback specifications.
- Static test cases.
- Read-only source inspection.

### Forbidden before
`required_specialist_fleet_complete_for_final_alakazam_v1`

- Model instantiation, even as a “small canary.”
- Parameter counting by loading a checkpoint.
- Checkpoint copying, conversion, widening, or migration.
- Replay scans or search-label materialization.
- Corpus construction, featurization, or cache warming.
- Gameplay generation.
- Training or rehearsal.
- Latency, memory, throughput, package, or parity benchmarks.
- Full-log decision replay execution.
- Package construction.
- Submission construction or upload.
- Service, selector, registry, environment, queue, or controller changes.
- Reserving CPU, GPU, RAM, or I/O that the old-system fleet can use.

Static arithmetic in this document is not model computation.

After that receipt and the resource lease pass, only the fixed Alakazam bridge
lane may perform model work. The two/three-archetype screen, S1, optional
turn-order research, ladder reconstruction, release packaging, and any other
capacity arm remain blocked until their own later gates.

## 5. Zero-disruption resource and mutation isolation

Zero disruption cannot honestly be guaranteed by merely lowering process
priority on shared active hardware.

After Slowking completion, and before the Alakazam final-format bridge starts,
require a second immutable receipt:

```text
capacity_research_resource_lease_v1
```

It must prove one of:

1. dedicated accelerator, CPU, RAM, storage-I/O, output paths, and process
   namespace; or
2. an exclusive post-fleet research window during which the production compute
   lane is quiescent and neither the Marnie's Grimmsnarl ex refresh nor
   population training has started.

If another production refresh or population training overlaps, the
final-format Alakazam/H10 work must use physically isolated capacity. If
neither condition is true, it remains staged.

Production remains the sole writer for:

- the canonical selector;
- specialist, cumulative-core, frozen, and runtime registries;
- `state/specialists.yaml`;
- the transition graph;
- managed production services and their handoff chains;
- submission authorizations; and
- the Kaggle queue.

The research lane must use:

- a separate root and output namespace;
- new derivative identities;
- content-addressed read-only parent copies;
- separate caches, logs, optimizer state, packages, and service names;
- no writable symlink or hardlink into production;
- no global parameter-limit override; and
- no automatic promotion hook.

Verify every parent digest before and after migration. A successful research
candidate remains offline until a later explicit release decision.

## 6. Keep version namespaces separate

Never overload a single `V` number.

Record these independently:

```text
core_generation
adapter_format
strategic_head_schema
decision_fusion_schema
capacity_profile
search_schema
research_derivative
```

Example:

```text
archetype: <selected-id>
core_generation: <accepted ancestry>
adapter_format: <parent format>
capacity_profile: H10-I/v1
search_schema: none | S1/v2
learned_route_inventory: sha256:<digest>
learned_route_count: <N>
guide_runtime_route_count: 0
parent_checkpoint: sha256:<digest>
```

Add the final base learned-route inventory digest and count to this record. The
search schema for a newly prepared S1 arm is `S1/v2`; `S1/v1` names only the
superseded historical scalar-column sketch in this document.

H10-I is a capacity recipe, not “core V12.” S1 is a search schema, not an
adapter version, and neither number replaces Training Core Revision, Matchup
Router Format, or Accepted Policy Generation.

## 7. Late-bound parent selection

Hammer-Pult is a historical architecture reference, not a universal weight
source.

### Mandatory Alakazam bridge parent

For the Revision-64 bridge, first lock the immutable historical Alakazam
passing checkpoint
`sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea`
as the preferred parent. Attempt direct migration only if its exact
model-config, feature, option, temporal, head, target, fusion, and adapter
contracts can be mapped completely and the migration can satisfy the
step-zero gates. New final-format structures must use declared zero-safe
initialization; an absent inherited key may not be filled from another
lineage.

If direct compatibility fails:

1. preserve a checksum-bound failure receipt naming every incompatible or
   missing contract;
2. resolve the newest checksum-accepted cumulative core;
3. create and normally validate a new same-archetype Alakazam refresh from
   that core under the then-current training contract; and
4. use only that new Alakazam derivative as the parent for final-format
   expansion.

The fallback is explicit and same-archetype before expansion. It never combines
historical Alakazam tensors and generic-core tensors in one partially
transplanted child.

### Broader finalist parents

For each later owner-provided archetype, resolve one exact parent only after
the applicable post-refresh completion receipt:

- it belongs to that exact archetype and deck;
- it is immutable and registered;
- it carries the final compatible old-system architecture contract;
- it has valid guide, corpus, fusion, adapter, and runtime evidence;
- it is eligible under locked local evidence; and
- it is not selected merely because it has the newest timestamp or largest
  core version number.

The parent lock must bind:

- archetype and deck digest;
- checkpoint digest and status;
- run, iteration, and lineage;
- accepted cumulative-core ancestry;
- complete serialized model-config digest;
- feature, option, temporal, head, target, and fusion schemas;
- the complete learned-head role map and distinct route inventory;
- adapter format and roster digest;
- guide and corpus digests;
- learned-parameter count and package size;
- trainer/code revision; and
- parent-selection evidence.

A newer generic cumulative core is ancestry, not a tensor overlay. It can
become a parent only through a separately registered same-archetype
old-system derivative.

### Same shapes, different weights

Shape compatibility proves migratability; it does not prove identity.

Construct each expanded child from its own resolved parent. Copy and clone only
that parent's tensors. Never take values from Hammer-Pult, another archetype,
or another cumulative core.

The migration test must expand at least two same-shape checkpoints whose
weights deliberately differ:

- child A reproduces parent A;
- child B reproduces parent B;
- copied tensors retain their respective parent values; and
- neither child refers to a hard-coded Hammer digest.

## 8. H10-I model card

### Purpose

Test whether a much larger inherited per-deck model can convert the stronger
completed-fleet curriculum into higher gameplay strength without rebuilding a
core from scratch.

### Architecture

```text
d_model                         96
attention heads                  8
spatial layers                   7
temporal layers                  3
option layers                    7
FF width                       2496
history context                 320
fusion width                     16
base learned decision heads       N, resolved from the final role map
distinct bounded action routes    N, exactly one per learned head
setup_board_outcome               included in N
validated Slowking combo head     included in N when its gate adds one
current-deck guide routes          0
```

At minimum, the successor schema carries the 17 historical learned decision
views plus `setup_board_outcome`, so `N >= 18`. If Slowking's revision-58
coverage gate proves the existing inventory insufficient and validates a
typed combo-state head, that head also enters `N`. The final value cannot be
declared until the Slowking and post-fleet role/route receipts are frozen.

Every learned head remains independently computed through its typed objective,
but none remains a diagnostic-only shadow. Its distinct route must combine its
typed output with the current legal-option representation after causal
board/state cross-attention, then contribute a finite bounded residual to the
action score. This includes `setup_board_outcome` and any Slowking combo-state
head. Fully removing a learned head from action influence is an ablation only,
never the default architecture. The guide is not a learned decision head and
is the sole no-route curriculum exception.

### Parameter accounting

The following arithmetic belongs only to the historical compatible d96,
`4/1/4`, FF384 parent with 17 fused inputs and 11 independently widened
strategic branches:

```text
historical H10-I added capacity = 8,396,722
historical H10-I total          = historical parent count + 8,396,722
```

For the historical roster18/V5 parent count:

```text
1,683,877 + 8,396,722 = 10,080,599
```

Different weights with identical shapes do not change that historical count.
The formula does not cover `setup_board_outcome`, a possible Slowking
combo-state head, or the distinct option-conditioned route schema required by
GOAL Revision 64.

After the Slowking fleet boundary, freeze the complete learned-head and route
inventory, then derive the actual H10-I module plan from it. The migration
receipt must name and count every inherited route, every widened branch, every
new residual branch, and every zero-safe gate. If the final parent carries a
different adapter format, vocabulary, fixed bank capacity, head set, or route
schema, instantiate and publish the actual total. Do not retain `8,396,722` or
`10,080,599` merely because the trunk is d96.

### Higher conditional arms

The research ceiling remains open:

| Arm | Historical profile only | Historical parameters | Use |
|---|---|---:|---|
| H10-I | d96, `7/3/7`, FF2496, independent heads | 10,080,599 | Primary |
| H12-I | d96, `8/3/8`, FF2816, independent heads | 12,293,143 | Only after positive H10 scaling |
| H15-I | d96, `9/3/9`, FF3328, independent heads | 15,530,903 | Later research only |

These are sizing references, not executable totals for the final role/route
inventory. Do not reduce H10 trunk capacity to make room for S1. Ten million
remains a research target floor, not a package quota; the exact count is a
post-fleet receipt.

## 9. Function-preserving H10-I inheritance

For each resolved parent:

1. Copy every compatible inherited fixed-shape embedding, policy, value,
   learned-head, per-head option-conditioned fusion-route, and
   matchup-adapter tensor exactly. The Alakazam bridge migration manifest must
   distinguish inherited tensors from genuinely new final-format structures;
   a missing inherited tensor is never sourced from a different lineage.
2. Preserve the first 384 FF neurons and their incoming rows, biases, and
   outgoing columns.
3. Add 2,112 FF neurons by cloning trained parent neurons.
4. Zero only the new outgoing columns so the wider FF computes the parent
   function at step zero.
5. Append three existing-type spatial blocks, two existing-type temporal
   blocks, and three existing-type option blocks.
6. Clone compatible internals from deep trained parent blocks.
7. Zero every appended block's attention output and second FF projection,
   including biases, so each new block begins as an identity.
8. Resolve the checksum-bound final head-role and route inventory. Add one
   `96 → 512 → output-width` residual MLP to every learned strategic head
   declared for H10 widening, including `setup_board_outcome` and any admitted
   Slowking combo-state head.
9. Zero each new branch's final projection and add its output to the unchanged
   original head output. Preserve its distinct inherited option-conditioned
   route and aggregate residual bound.
10. Copy the current-deck guide identity, curriculum schedule, source, and
    checksum bindings as training metadata only. There is no guide runtime
    tensor, fusion column, or action route to inherit.
11. Preserve the exact parent adapter interface. Do not combine an adapter
    migration with H10.
12. Record every source tensor, neuron clone index, appended-block source,
    learned-head role, distinct route, route bound, and zeroed output in a
    migration manifest.

For the preferred historical-Alakazam hot start, every genuinely new
final-format head/route or widened branch must be zero-safe at step zero. If
the remaining inherited model cannot reproduce the historical Alakazam
function within the declared parity bounds, direct migration is rejected and
the same-archetype fallback chain in Section 7 is mandatory.

Step-zero gates:

- exact legal masks;
- exact greedy actions;
- exact adapter route or bypass decisions;
- no missing or unexpected inherited keys;
- parent/child logits, values, and typed-head outputs within
  `atol=1e-5, rtol=1e-5`;
- exact one-to-one learned-head/route inventory identity;
- guide-metadata present and absent cases with identical runtime inputs;
- proof that guide metadata produces no runtime tensor or logit route;
- all applicable adapter-route cases;
- short histories; and
- a full 320-decision history.

The parent remains byte-for-byte unchanged.

## 10. Conditional learned search head S1

### Hard judgment

S1 is beneficial enough to plan, but not proven enough to bundle into H10-I.

The current model already owns action-Q, tactical-outcome,
opponent-response, resource-forecast, outcome-distribution, and
remaining-turns heads. A search head trained on the same terminal outcome would
be redundant.

S1 instead predicts an option-conditioned continuation advantage between the
current decision and the next decision by the same player.

### Target

For chosen option `a_t`, let `t+` be the next same-seat decision or terminal:

```text
y_S1 =
  clip(
    G[t:t+] + gamma^delta * V_parent(causal_prefix[t+])
    - V_parent(causal_prefix[t]),
    -1,
    1
  )
```

Rules:

- `V_parent` is the frozen exact parent teacher.
- Terminal continuations use zero next value.
- Incomplete, censored, ambiguous, or unverifiable transitions are masked.
- Future events are used only to construct a training target.
- Runtime inputs contain only the causal current prefix and current legal
  option representation.
- Unchosen counterfactual actions receive no invented label.

This is a lower-variance, option-specific continuation target. It is distinct
from terminal Monte Carlo action-Q and the state-conditioned future heads.

### Architecture and late-bound size

```text
option hidden 96 → 512 → 1
```

The S1 prediction MLP has a fixed count:

```text
head MLP = (96 × 512 + 512) + (512 × 1 + 1) = 50,177
```

The superseded scalar-column sketch had this historical arithmetic:

```text
historical scalar fusion column       48
historical total S1 addition      50,225
```

That 48-parameter column is not an authorized S1/v2 route. Let:

```text
N_base  = checksum-bound learned-route count in the frozen H10-I parent
P_route = instantiated parameter count of one distinct bounded
          option-conditioned S1/v2 route under the final fusion schema
```

Then:

```text
S1 route ordinal = N_base + 1
S1 addition      = 50,177 + P_route
H10-I+S1 total   = instantiated H10-I count + 50,177 + P_route
```

The migration receipt must publish `N_base`, the resulting ordinal, the route
schema digest, `P_route`, and the exact total. It may not reuse
`10,130,824` or `50,225` as a final count merely because those values appeared
in the historical sketch.

### Integration

- S1 is the next learned decision route after the frozen base inventory; it is
  not hard-coded as the eighteenth input.
- Preserve every existing distinct per-head route exactly.
- Give S1 its own bounded option-conditioned route that combines the S1 typed
  output with the current legal-option representation after causal
  board/state cross-attention.
- Initialize S1's route gate/output to zero so step-zero logits and actions
  match H10-I exactly while retaining the final aggregate residual bound.
- Clone S1's first projection from the H10-I action-Q residual branch.
- Initialize S1's final projection and bias to zero.
- An S1-absent checkpoint has an exact versioned route-inventory bypass.
- Reject any implementation that appends S1 as an option-blind scalar fusion
  column or merges it into another head's route.
- Runtime remains one ordinary forward pass followed by legal mask and greedy
  action.

There is no tree, node expansion, UCB, simulator call, rollout, or repeated
inference.

### Training-system boundary

S1 necessarily adds one model output, one target slot, one masked loss entry,
and one distinct bounded option-conditioned route. It does not require a new
trainer, optimizer, cadence, replay algorithm, or policy loop.

The intended implementation reuses:

- the existing typed-head target schema pattern;
- masked Smooth-L1;
- a fixed initial weight of `0.05`;
- the existing bootstrap stage alongside action-Q; and
- ordinary RL and rehearsal updates.

If “do not change the training system” is later interpreted as prohibiting even
one versioned typed target and loss slot, S1 cannot honestly exist and must be
omitted. H10-I remains valid without it.

### Admission gates before S1 training

All S1 computation waits for the fleet-complete and resource receipts.

Require:

1. at least 95% accounted coverage over eligible complete decisions;
2. whole-game-disjoint train, validation, and locked sets;
3. no private or future feature leakage;
4. at least 5% held-out predictive improvement over the best reconstruction
   from existing typed outputs;
5. useful calibration and selected-versus-runner-up ranking; and
6. a matched H10-I control;
7. exact step-zero H10-I parity through a zero-safe S1 route;
8. legal-option dependence and causal suffix-invariance;
9. nonzero finite post-training S1 leave-one-route-out logit influence; and
10. one-to-one binding of S1 to the next ordinal in the role/route receipt.

If distinctness fails, reject S1 before expensive training.

## 11. Optional going-second training

Turn-order optimization is not part of the base H10-I architecture experiment.
The default is:

```text
turn_order_training: canonical
```

Every base arm inherits the then-current canonical old-system game-generation
and evaluation schedule unchanged.

The mandatory post-Slowking Alakazam refresh is stricter: its assigned and
consumed training seats must be exactly 50% first and 50% second, its eventual
package preference is `first_if_allowed`, and it is ineligible for
`second_focus_1_to_7`, an always-second behavior branch, or a
second-preferring package under this refresh.

After an archetype's H10-I architecture is frozen, the owner may enable:

```text
turn_order_training: second_focus_1_to_7
```

That creates a separately named derivative and imports the exact training
contract from `docs/TOP_100_TURN_ORDER_OPTIMIZATION_PLAN.md`:

```text
candidate first : candidate second = 1 : 7
```

Required controls:

- the same H10-I parent;
- the same architecture;
- a canonical-seat continued H10-I control;
- matched opponent versions, seeds, interactions, and optimizer exposure;
- separate assigned, actual, and consumed curriculum receipts; and
- evaluation that is not sampled at the training ratio.

The optional arm answers one question only:

> Does second-focused training improve the intended second-seat distribution
> without making the underlying H10 capacity result uninterpretable?

It does not authorize an always-second package. Packaging preference remains a
separate later decision.

To contain compute, test second-focused training first on the leading
archetype. Expand it to a second archetype only after a positive matched result.

## 12. Existing training cadence

After the 25-epoch supervised bootstrap, use the current ordinary cadence:

For the mandatory Alakazam bridge, every bootstrap/RL/rehearsal game-generation
receipt that owns seat assignment must prove an exact even first/second split.
Evaluation remains separately balanced and training-ineligible.

| Block | Cumulative training | Fresh games per arm | Decision |
|---|---:|---:|---|
| A | 5 RL + 5 rehearsal | 40,960 | Stability, throughput, and early gameplay |
| B | 10 RL + 10 rehearsal | 81,920 | Require material matched improvement |
| C | 15 RL + 15 rehearsal | 122,880 | Seed and locked-panel confirmation |
| D | 20 RL + 20 rehearsal | 163,840 | Lead archetype only while validation improves |

The practice fleet is frozen inference-only opposition. It is not averaged
into the student.

For each archetype, resolve one parent `P` and construct:

```text
P0         untouched immutable parent
P1         continued-current-size control
H10-I      function-preserving capacity expansion
H10-I+S1   conditional search arm
H10-I-2nd  optional second-focused derivative
```

All matched arms use the same parent digest, corpus, opponent snapshot, game
schedule, seeds, and evaluation sets. Never compare a new-core H10 child with a
continued Hammer or older-core control.

Select the best immutable checkpoint, not the last epoch.

## 13. Gate sequence

### G-1 — Static staging

Complete this document, formulas, role/route schemas, receipt templates,
non-runnable package manifests, and test specifications. This is the only
currently authorized implementation-preparation stage. No checkpoint load,
model instantiation, replay scan, data materialization, or model computation.

### G0 — Two completion barriers

`G0-A` requires
`required_specialist_fleet_complete_for_final_alakazam_v1`. It proves all 15
required specialists frozen, including the exact remaining Archaludon ex then
Slowking order and every cumulative disposition through Slowking. The preserved
inference-only Crustle opponent and matchup route do not count as a specialist.
Together with G1, G0-A authorizes model work only for the mandatory Alakazam
final-format refresh.

`G0-B` requires `post_refresh_sequence_complete_for_capacity_v2`. It proves
that the final-format Alakazam passed, froze, and registered, the following
Marnie's Grimmsnarl ex refresh completed, and every intervening cumulative
disposition is terminally resolved. G0-B authorizes the later
two/three-archetype capacity program.

No applicable receipt, no model work.

### G1 — Resource and write isolation

Require `capacity_research_resource_lease_v1` and a writable-path audit.

No isolated lease, no model work.

### G2 — Parent lock

For the G0-A bridge, lock exactly `alakazam` and produce one
`final_format_alakazam_parent_lock_v1`. Prefer the immutable historical
Alakazam checkpoint; otherwise require the preserved direct-migration failure
and exact same-archetype fallback chain from Section 7.

For the later G0-B program, validate exactly two or three owner-provided
archetypes and produce one `capacity_parent_lock_v1` per archetype.

In both cases, freeze the complete learned-head role/route inventory and prove
exactly one distinct bounded option-conditioned route per learned head, with
zero guide routes.

### G3 — H10-I migration

Create research-only derivatives. Require step-zero parity and exact
instantiated module/parameter receipts. Historical 17-input, 11-branch, and
parameter totals are not accepted as substitutes for the late-bound inventory.

### G4 — Feasibility canary

Measure:

- cold start;
- p50 and p95 CPU decision latency;
- peak CPU and accelerator memory;
- training throughput;
- complete-game completion;
- invalid actions;
- timeouts; and
- package size.

The parameter-limit override is research-run-local.

### G5 — Base capacity screen

Compare P0, P1, and H10-I.

- The mandatory Alakazam bridge uses exact balanced 50/50 training seats,
  package preference `first_if_allowed`, and no second-focused arm. It must
  pass its normal gameplay, runtime, freeze, and registration gates before the
  Marnie's Grimmsnarl ex refresh begins.
- With two inputs, both may reach Block B.
- With three inputs, all reach Block A and at most two reach Block B.
- Continue only when added capacity receives meaningful gradients and updates.

### G6 — S1 admission and screen

Materialize labels, run the distinctness probe, and train H10-I+S1 only after
H10-I itself is stable.

S1 is not part of the mandatory Alakazam refresh unless a later owner decision
explicitly authorizes it after the base final-format Alakazam is frozen.

Require S1/v2 to occupy the next role/route ordinal through its own bounded
option-conditioned route. A scalar fusion column fails this gate. S1 failure
falls back to H10-I and never invalidates a successful capacity result.

### G7 — Optional going-second screen

If requested in the input manifest, create `H10-I-2nd` only after base H10-I is
frozen. Begin with the leading archetype.

The mandatory Alakazam refresh is excluded from G7 by Revision 64.

### G8 — Confirmation

Use at least three training seeds, frozen opponent versions, out-of-time data,
unseen deck variants, and a locked release panel.

### G9 — Ladder reconstruction

Fit the local-to-Kaggle proxy using immutable historical submissions and frozen
local matrices. Do not adapt model weights from the result.

### G10 — Decision-replay fidelity

Run the offline decision-replay and expert-review views only against immutable
candidate copies. Require exact ordinary/instrumented inference parity,
per-head route reconstruction, causal suffix-invariance, and complete
leave-one-route-out attribution. Guides appear only as retrospective training
metadata, never as a runtime tensor or logit component.

### G11 — Offline package

Create a separately named, non-submittable research package and validate CPU
parity, exact model/config/role/route/deck digests, latency, memory, and package
size. The package manifest must say `kaggle_eligible: false` and have no queue
authorization.

### G12 — Offline release recommendation

Freeze the gameplay, capacity-use, replay-fidelity, ladder-proxy, and offline
package evidence in one checksum-bound recommendation. A recommendation is not
release authority.

### G13 — Separate owner release decision

Selector eligibility, registry promotion, production packaging, and Kaggle
submission require a new explicit owner authorization bound to the exact
candidate and evidence set. They are not automatic effects of this plan.

### G14 — Authorized production package and submission preflight

This gate remains blocked until G13 grants exact authority. Then, and only
then, construct the production package, verify checkpoint/deck/config/route
digests and turn-order behavior, mint the checksum-bound one-shot submission
authorization, and hand it to the asynchronous queue.

The queue's four-hour spacing anchor is the second-most-recent logical
submission after deduplicating reconciled Kaggle and local queue rows by
submission ID or checkpoint-bound label. The newest logical submission alone
never delays the next copy; with fewer than two prior logical submissions
there is no spacing anchor. Daily quota, oldest-first order, one-shot identity,
and non-blocking training behavior remain unchanged.

## 14. Local strength formulas

The repository has two complementary local strength formulas. Preserve both.

### Connected Elo ladder

The current connected ladder implementation is `scripts/rank_baselines.py`.

Its primary rating starts every agent at:

```text
R_0 = 1500
K   = 20
passes over recorded games = 8
```

For agent A against B:

```text
E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
R_A' = R_A + K * (S_A - E_A)
```

Decisive games update both players. The current code handles draws as two
symmetric half-score entries with a half-K step.

The secondary local measures are:

```text
point_WR_i = points_i / games_i

SoS_i =
  sum_j(games_ij * point_WR_j)
  / sum_j(games_ij)

SoS_adjusted_i = point_WR_i * SoS_i
```

The implemented rank order is:

```text
1. Elo descending
2. overall point win rate descending
3. SoS-adjusted score descending
4. stable agent ID
```

Every post-fleet reconstruction must freeze:

- agent/version identities;
- pairwise game records and order;
- seed schedule;
- seat schedule;
- Elo constants and pass count;
- opponent snapshot;
- point/draw convention; and
- the exact code digest.

Changing the formula creates a new proxy version and cannot rewrite old
predictions.

The implementation is order-dependent because it replays recorded games for
eight passes. Freeze the canonical game order. For uncertainty, resample
matched seat-pair clusters and preserve the sampled ordering within each
bootstrap replicate.

### Premium fixed-field strength

The current premium field aggregation is implemented in
`poke_bot/pure_rl/strong_public_gate.py`.

For opponent `i`, let `mean(score_i)` be the mean over matched seat-0/seat-1
clusters and let `w_i` be its frozen field weight:

```text
premium_weighted_WR =
  sum_i(w_i * mean(score_i))
  / sum_i(w_i)
```

The existing gate:

- clusters adjacent games as one seat-0/seat-1 pair;
- resamples within each opponent rather than pooling all games;
- uses 4,000 bootstrap resamples by default;
- reports a 90% interval in the current gate contract;
- uses 250 games per opponent, 125 per seat, in the current program; and
- currently gives S/S+ opponents weight 2 and A opponents weight 1.

These values describe the current implementation, not an eternal research
constant. The post-fleet proxy must bind the exact frozen roster, weights,
sample count, confidence level, and code digest it actually uses.

Historical premium-gate values from different roster revisions are not
directly comparable. Recompute every candidate against one common post-fleet
field.

Use Elo to place all agents on a connected relative ladder. Use premium
weighted WR and the complete matchup vector to measure relevance to the field
we actually expect.

## 15. Kaggle ladder reconstruction

### What is publicly known

The competition describes each submission's skill as:

```text
skill ~ Normal(mu, sigma^2)
```

New submissions begin at `mu = 600`. Matchmaking favors similarly rated
agents. Wins increase `mu`, losses decrease it, draws pull ratings toward one
another, and uncertainty decreases as evidence accumulates. New submissions
receive games more frequently. Only the latest two submissions are active, and
the team leaderboard displays the better score.

The exact update constants and private opponent ratings are not published.
Therefore this project is an emulator with uncertainty, not a claim to recover
the exact Kaggle state.

Official evaluation description:
<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description>

### Required immutable observation table

For every historical submission, record:

- Kaggle submission ID;
- upload label;
- submission and checkpoint digests;
- archetype and exact deck digest;
- model/core/adapter/head/search versions;
- timestamp uploaded;
- every timestamped observed public skill score;
- every timestamped observed rank;
- active/inactive state;
- known games played, if exposed;
- validation/error state;
- packaging turn-order behavior;
- local-ladder Elo, point-WR, SoS, worst-matchup, and reliability values from
  the predeclared frozen matrix;
- premium weighted WR and its opponent-stratified interval;
- local evaluation timestamp and opponent-snapshot digest; and
- whether the checkpoint was a measured pass, owner accepted, or
  ceiling-accepted.

Never collapse two submissions merely because they used the same deck.
Repeated copies of one checkpoint remain one checkpoint group; their separate
score trajectories estimate convergence and observation noise rather than
acting as independent training examples.

Use this evidence precedence:

```text
1. direct timestamped Kaggle row plus immutable attempt receipt
2. canonical reconciled submission-queue row
3. historical specialist record
4. compatibility or dashboard projection
```

Conflicting lower-precedence values remain recorded as stale observations; they
never override an immutable or direct source. A score exactly at the initial
600 without a trustworthy age or game count is treated as unconverged/censored
evidence, not a stable skill label.

### Layer A — Local strength model

For each of the two or three candidate archetypes:

1. Run a frozen round-robin against the same complete practice fleet.
2. Compute the exact implemented local Elo ranking.
3. Preserve the complete pairwise matrix.
4. Add meta-weighted point-WR, macro-opponent WR, worst-decile value,
   premium weighted WR, reliability, latency, and uncertainty.
5. Preserve balanced 50/50, actual-first, and actual-second diagnostic views
   without changing the default proxy-selection view.
6. Bootstrap matched whole-game seat-pair clusters to obtain a distribution
   rather than one Elo number.

### Layer B — Local-to-Kaggle calibration

Begin with a low-capacity bridge. For candidate `i` and field opponent `j`:

```text
p_ij = 1 / (1 + 10 ** (-(R_i - R_j) / 400))

field_p_i =
  sum_j(w_j * p_ij)
  / sum_j(w_j)

field_delta_i =
  400 * log10(field_p_i / (1 - field_p_i))
```

The `400` scale belongs to the local Elo bridge. It is not a claim that Kaggle
uses the same scale.

Fit:

```text
predicted_mu_i,t =
    calendar_intercept_t
  + beta * field_delta_i
  + convergence_curve(age_or_games_i,t)
  + checkpoint_group_error_i
```

Use a robust monotone or hierarchical fit grouped by checkpoint digest. When
the stabilized submission sample is small, prefer Theil-Sen or monotone
isotonic calibration to an archetype-specific multi-feature regression.

Premium weighted WR, SoS, lower-tail value, reliability, and meta mix remain
diagnostics and sensitivity variables. Add one only when checkpoint-held-out
evidence proves it improves calibration.

Observed Kaggle scores are noisy and convergence-dependent. Use strong
regularization and group successive copies of one checkpoint.

### Layer C — Approximate rating dynamics

Fit a Gaussian/TrueSkill-like update family to our timestamped score
trajectories:

- initial `mu = 600`;
- an inferred initial uncertainty;
- win/loss/draw update magnitude;
- uncertainty decay;
- new-submission game-rate multiplier; and
- score-convergence rate.

If per-game traces are unavailable, fit aggregate score transitions and widen
the interval. Do not fabricate opponent identities or exact sigmas.

### Layer D — Synthetic public ladder

For each candidate:

1. Draw local strength from its whole-game bootstrap.
2. Map it through the posterior local-to-Kaggle calibration.
3. Initialize a synthetic submission at 600 with fitted uncertainty.
4. Match it primarily against nearby synthetic ratings.
5. Draw opponent archetypes from timestamped top-ladder replay prevalence.
6. Draw game outcomes from the candidate's frozen local pairwise model.
7. Apply the fitted approximate Gaussian rating update.
8. Repeat through the expected remaining evaluation horizon.
9. Insert the simulated score into a timestamp-matched public score
   distribution.
10. Repeat enough times to obtain rank and score intervals.

Outputs per candidate:

- predicted converged `mu` median and 50/80/95% intervals;
- predicted current and final rank distributions;
- probability of stabilized score above 800 and above 1000;
- probability of top 100, top 50, top 20, top 10, and first;
- matchup contributions to rating;
- probability that another provided archetype outranks it;
- expected best-of-two team score for proposed active pairs; and
- sensitivity to meta mix, convergence time, and rating-update assumptions.

For two archetype inputs, simulate the one available active pair. For three
inputs, simulate all three possible active pairs:

```text
(A, B)
(A, C)
(B, C)
```

Because Kaggle displays the team's better active submission, report both the
expected maximum score and the downside correlation of each pair. A pair of
near-identical candidates may have a higher mean but less hedge value than two
archetypes whose matchup failures differ. This analysis recommends a pair; it
does not submit or activate one.

### Public score-to-rank curve

When timestamped public leaderboard snapshots are available, fit a monotone
empirical curve:

```text
rank_t(mu) =
  1 + count(public_scores_t > mu)
```

If only sparse score/rank pairs are available, fit a monotone quantile spline
and report much wider intervals. Never extrapolate a precise top rank from two
or three team observations.

### Validation

Use:

- leave-one-checkpoint-out prediction, keeping every copy and snapshot of the
  same checkpoint in one fold;
- forward-chaining time splits;
- held-out archetype tests where possible;
- mean absolute Kaggle-score error;
- median absolute rank error;
- Spearman and Kendall ordering;
- top-100 and top-20 calibration;
- interval coverage;
- ablation of Elo, SoS, meta weighting, tail, and reliability terms; and
- a naive baseline using Kaggle `mu=600` or local Elo alone.

Suggested credibility gate:

- leave-one-checkpoint-out Spearman at least `0.70`;
- stabilized-score MAE at most 75 rating points;
- 80% prediction-interval coverage between 70% and 90%; and
- positive held-out pairwise ordering accuracy versus Elo alone.

If the sample is too small to evaluate those conditions, expose only a local
strength index and broad scenario intervals. Do not publish a projected Kaggle
score.

The proxy is admissible for finalist selection only if it ranks held-out
submissions usefully and its intervals are calibrated. Otherwise use the
locked local panel directly.

### Anti-overfitting rules

- Kaggle public score never enters training targets, replay, or gradients.
- Do not use a candidate's own Kaggle result to select that same checkpoint.
- Calibration data ends before the locked finalist evaluation.
- Repeated snapshots of one submission are correlated and remain grouped.
- Team-best display, active-two policy, submission age, and meta date remain
  explicit.
- A 600 score may mean “new or under-observed,” not necessarily “weak.”
- A ceiling-accepted checkpoint remains labeled as such.
- Refit versions are immutable and time-stamped.

## 16. Promotion gates

### H10-I early continuation

- approximately +1 point on the locked fixed-mix gameplay measure versus P1;
- no reliability regression;
- no critical matchup more than 3 points worse;
- feasible completion time;
- acceptable memory and serving cost; and
- meaningful gradients and updates in added capacity.

### H10-I final promotion

- at least +2 points versus untouched P0;
- at least +2 points versus matched continued P1;
- positive paired 95% lower confidence bound;
- at least two of three seeds positive;
- no seed worse than -1 point;
- no critical matchup more than 3 points worse;
- no worst-decile collapse;
- no invalid-action, timeout, OOM, or complete-game regression;
- out-of-time and unseen-deck persistence; and
- added-capacity ablation removes at least half the measured gain.

### S1 promotion over H10-I

- target distinctness and label gates pass;
- at least +1 point over matched H10-I;
- positive paired 95% lower confidence bound;
- removing S1 eliminates at least half its incremental gain;
- a duplicate-action-Q control does not reproduce the gain;
- shuffled S1 targets do not reproduce the gain;
- no critical-matchup or reliability regression; and
- no more than 2% measured p95 CPU decision-latency overhead.

### Optional second-focused arm

Its detailed promotion contract remains owned by the turn-order plan. At
minimum it must beat a matched canonical-curriculum H10 control on its declared
second-seat target distribution without an unacceptable general-strength,
matchup, or reliability loss.

### Ladder-proxy admission

- leave-one-checkpoint-out Spearman at least 0.70;
- stabilized-score MAE at most 75 rating points;
- 80% interval coverage between 70% and 90%;
- better held-out pairwise ordering than local Elo alone;
- stable candidate order under plausible meta and update assumptions; and
- no use of same-candidate Kaggle outcomes for checkpoint selection.

## 17. Capacity-use proof

Nominal size is not evidence that H10 learned.

Record:

- gradient and update norms for added FF channels;
- gradient and update norms for appended spatial, temporal, and option blocks;
- gradient and update norms for every independently computed learned branch,
  including `setup_board_outcome`, any admitted Slowking combo-state head, and
  S1 when present;
- gradient and update norms for every distinct option-conditioned action route;
- activation variance and effective rank;
- cloned-channel correlation;
- label coverage and calibration for every strategic head; and
- per-head legal-option dependence and leave-one-route-out logit/action
  attribution; and
- S1 coverage, calibration, route bound, and attribution when present.

Then:

- disable all new capacity;
- ablate each appended block group;
- ablate the third temporal layer;
- ablate each strategic residual branch;
- ablate each learned head's action route independently;
- prune 25% and 50% of added FF channels;
- zero S1's bounded route;
- remove S1 entirely; and
- repeat the locked gameplay panel.

If added capacity can be removed without removing at least half the gain, the
extra parameters are dead, redundant, or merely regularizing.

## 18. Decision replay and expert review

Before fleet completion, preserve only this static specification. Do not
instrument production.

After architecture selection, replay a complete public game log against an
offline checkpoint copy and stop at a selected decision's causal prefix.

Required views:

- decision timeline;
- legal options;
- base logits, every named per-head route residual, S1 route residual when
  present, aggregate bounded residual, and final logits;
- selected-versus-runner-up margin;
- value and all typed-head outputs;
- leave-one-head/route-out and leave-S1-route-out effects;
- board and history attribution;
- active matchup route or bypass;
- the frozen role/route inventory and bound applied at that decision;
- guide identity and curriculum evidence in a visibly training-only metadata
  panel with no runtime value or logit component;
- public-state counterfactuals;
- parameter/update health by module; and
- later events in a visibly retrospective, non-causal panel.

Fidelity gates:

- instrumented and ordinary inference produce identical logits and choices;
- base plus all named bounded route residuals reconstruct final pre-mask
  logits;
- every learned head has one independently ablatable route and no learned head
  is diagnostic-only;
- omitting guide metadata leaves runtime logits byte-identical;
- changing a future log suffix cannot change an earlier trace;
- no private information enters the model or explanation; and
- perturbation tests support claimed attributions.

Experts correct actions, acceptable sets, rankings, and reasoning. They do not
edit tensor values.

## 19. Boundary-relative schedule

Let:

```text
A0 =
  15-specialist fleet-complete receipt through Slowking
  + required cumulative-core dispositions through Slowking
  + valid resource lease

T0 =
  A0
  + final-format Alakazam refresh completion
  + Marnie's Grimmsnarl ex refresh completion
  + all intervening cumulative-core dispositions
  + post-refresh capacity receipt
```

No model action precedes A0. Between A0 and T0, only the mandatory Alakazam
final-format bridge is authorized.

Relative order:

| Window | Work | Exit |
|---|---|---|
| A0 | Lock historical Alakazam; validate direct migration or record and build the same-archetype latest-core fallback | Exact Alakazam ancestry |
| A0 + 1 | Final-format Alakazam migration and step-zero parity | Behavior-preserving Alakazam child |
| Next | Exact 50/50-seat Alakazam bootstrap, matched controls, feasibility, and gameplay gates | Passed/frozen/registered Alakazam refresh |
| Next | Marnie's Grimmsnarl ex refresh and intervening cumulative dispositions | `post_refresh_sequence_complete_for_capacity_v2` |
| T0 | Validate two/three inputs and lock broader finalist parents | Exact ancestry and controls |
| T0 + 1 | H10 migration and step-zero parity | Behavior-preserving children |
| Next | Feasibility canary and 25-epoch bootstrap | Runnable isolated candidates |
| Next | Block A for all two/three archetypes | Prune failures |
| Next | Block B for at most two | Select lead architecture/deck |
| Next | S1/v2 admission and matched S1 screen on lead | Keep H10 or H10+S1 |
| Next | Optional second-focused screen on lead | Keep canonical or second-focused derivative |
| Next | Three-seed confirmation and locked panel | Immutable finalists |
| Next | Ladder emulator and decision-replay fidelity | Interpretable locked evidence |
| Next | CPU and non-submittable package validation | Offline release recommendation |
| Final | Separate owner release decision | Production package/submission only if exactly authorized |

Do not begin an arm unless measured post-T0 throughput shows it can finish
complete training, confirmation, and packaging with at least a 72-hour
competition buffer.

If fleet completion leaves insufficient time, submit the strongest old-system
model. Do not compress validation, borrow production hardware, or interrupt the
fleet to manufacture an H10 window.

## 20. Hard forecast

These are subjective priors, not measured evidence.

- H10-I remains the highest-value model-size bet.
- For one well-selected archetype, the chance H10-I beats matched extra
  training by at least two gameplay points is roughly 40–45%.
- Screening two strong archetypes raises the correlated chance to roughly
  55–60%.
- Adding a third archetype improves search coverage, but less than an
  independent third trial because architecture and curriculum risks are
  shared.
- S1 has roughly a 20–30% chance of adding at least one further gameplay point
  over H10-I. Its most likely positive gain is smaller, around 0.2–0.7 points.
- S1's main risk is target redundancy, not its fixed 50,177-parameter
  prediction MLP. The required S1/v2 route and exact total remain late-bound.
- A second-focused derivative may add ladder value, but must be measured after
  architecture selection so it does not hide whether H10 itself worked.
- The ladder emulator can improve ordering and risk estimates, but our
  submission sample will remain small; it should output broad intervals.
- The chance that expansion, extra epochs, S1, and finalist selection together
  supply the entire approximate 800-to-1000 jump remains around 20–25%.

My hard bet:

> Complete all 15 specialists in the exact remaining Archaludon-then-Slowking
> order, immediately build the preferred-old-parent/fail-closed-fallback
> Alakazam refresh in final-submission format with exact 50/50 training seats
> and a first-if-allowed package, finish the following Grimmsnarl refresh,
> input the strongest two or three archetypes, run H10-I against matched
> controls, test S1 only on the later lead, optionally add the separate
> second-focused derivative only to eligible non-bridge arms, and use the
> calibrated ladder proxy as a final ranking aid rather than as the optimizer.

## 21. Stop and rollback

Stop an arm when:

- it cannot finish with the validation buffer;
- two capacity steps fail;
- matched extra-training explains the gain;
- one seed or one opponent drives the result;
- a critical matchup or lower tail collapses;
- runtime reliability deteriorates;
- S1 duplicates existing targets;
- S1 fails coverage or distinctness;
- the going-second arm harms its declared utility;
- the ladder proxy fails held-out ordering;
- explanation fidelity fails; or
- production isolation is lost.

Rollback requires no production reversal:

- parents remain immutable;
- failed derivatives receive rejected research receipts;
- S1 failure returns to H10-I;
- H10-I failure returns to P1 or P0;
- going-second failure returns to canonical H10;
- ladder-proxy failure returns to the locked local panel; and
- no selector, service, registry, or queue restoration is required because the
  research plan never changed them.

## 22. Required research records

- Required-specialist fleet completion-through-Slowking receipt.
- Resource lease and writable-path audit.
- Immutable historical Alakazam preferred-parent lock.
- Alakazam direct-migration compatibility receipt, or its preserved failure
  plus exact same-archetype latest-core fallback-chain receipts.
- Exact 50/50 assigned/actual/consumed Alakazam training-seat receipts.
- Final-format Alakazam pass, freeze, and registration receipt with
  `first_if_allowed` packaging metadata.
- Post-refresh sequence completion receipt after Marnie's Grimmsnarl ex.
- Two/three-archetype owner input manifest.
- Frozen practice-fleet snapshot.
- Per-archetype parent locks.
- Final learned-head role map and exact one-route-per-learned-head inventory.
- Zero-route guide curriculum-metadata manifest.
- Migration manifests.
- Step-zero parity receipts.
- Exact learned-parameter and package counts.
- P0/P1/H10 matched screens.
- S1/v2 target, coverage, distinctness, route-ordinal, calibration, and
  ablation receipts.
- Optional going-second curriculum and evaluation receipts.
- Whole-game bootstrap and locked-panel identities.
- Local ladder matrix and exact formula/code digest.
- Historical Kaggle submission observation table.
- Local-to-Kaggle calibration version.
- Synthetic ladder assumptions and posterior rank distributions.
- Leave-one-submission-out proxy validation.
- Decision-replay fidelity receipts.
- Non-submittable offline package manifest and CPU parity receipt.
- Final offline release recommendation.
- Separate later production authorization, if any.
- Authorized production package and submission-preflight receipt, if released.
- Deduplicated queue snapshot and second-most-logical-submission spacing
  calculation, if submitted.

## 23. Review checklist

- [ ] Keep work static-only before the Slowking fleet boundary: schemas, templates,
      tests, and non-runnable manifests; no model or replay computation.
- [ ] Require all 15 canonical specialists, with Archaludon then Slowking as
      the remaining order.
- [ ] Immediately after Slowking, run the separately versioned Alakazam
      refresh as the first final-submission-format model.
- [ ] Prefer the immutable historical Alakazam checkpoint; on incompatibility,
      preserve the failure and use the exact same-archetype latest-core
      fallback path before final-format expansion.
- [ ] Never rewrite old Alakazam or partially mix its tensors with a generic
      core.
- [ ] Train the new Alakazam with an exact 50/50 seat split and package it
      `first_if_allowed`; do not apply the second-focused plan to this refresh.
- [ ] Require the following Grimmsnarl refresh and all core dispositions before
      broader multi-archetype capacity research.
- [ ] Keep Crustle out of specialist training and completion while preserving
      its exact public inference-only opponent and active matchup route.
- [ ] Require dedicated hardware or an exclusive post-fleet window.
- [ ] Accept exactly two or three archetypes for the later broad screen; the
      mandatory Alakazam bridge is the single-archetype exception.
- [ ] Late-bind one exact parent per archetype.
- [ ] Treat same-shaped different weights as distinct checkpoint identities.
- [ ] Use Hammer-Pult only as a historical system reference.
- [ ] Preserve the exact resolved adapter format.
- [ ] Bind one distinct bounded option-conditioned action route to every
      learned head, including setup and any admitted Slowking combo head.
- [ ] Keep the current-deck guide as training-only metadata with no runtime
      tensor, fusion route, or action-logit authority.
- [ ] Build H10-I through function-preserving inheritance.
- [ ] Run matched continued-current-size controls.
- [ ] Keep S1 conditional and single-pass; add it as the next learned route,
      never as an eighteenth scalar fusion column.
- [ ] Run no MCTS or simulator rollout search.
- [ ] Default turn-order training to canonical.
- [ ] Permit `second_focus_1_to_7` only as a later separate derivative.
- [ ] Do not let training preference automatically set package preference.
- [ ] Recreate the local ladder using the implemented Elo/WR/SoS formula.
- [ ] Calibrate to Kaggle using immutable historical submissions.
- [ ] Report Kaggle score and rank distributions, not false precision.
- [ ] Keep Kaggle observations out of gradients and replay.
- [ ] Promote only replicated gameplay gains.
- [ ] Require replay-fidelity and non-submittable package gates before a
      release recommendation.
- [ ] Require a separate explicit production release decision.
- [ ] After authorization, compute queue spacing from the deduplicated
      second-most-recent logical submission, never the newest alone.

## 24. Research references

- `GOAL.md`
- `docs/COMPLETE_FLEET_TOP_FINISH_PLAN.md`
- `docs/TOP_100_TURN_ORDER_OPTIMIZATION_PLAN.md`
- `docs/RL_TRAINING_PROTOCOL.md`
- `docs/PURE_RL_PIPELINE.md`
- `config/rl_protocol.yaml`
- `poke_bot/model.py`
- `poke_bot/pure_rl/model_profile.py`
- `poke_bot/pure_rl/strong_public_gate.py`
- `poke_bot/matchup_adapters.py`
- `scripts/rank_baselines.py`
- `scripts/ladder_meta_report.py`
- `ops/alakazam_gate_program_v1.json`
- `state/causal_decision_fusion_validation_v1.json`
- `state/matchup_adapter_roster.json`
- Kaggle competition evaluation description:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description>
