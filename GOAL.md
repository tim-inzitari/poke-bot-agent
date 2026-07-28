# Pokémon RL Goal Gateway

Schema: `poke_bot.goal_gateway/v1`  
Revision: `23`  
Status: `authoritative`

This file is the stable entry point for every resumed controller. The product
goal should remain short:

> Continuously execute the authoritative contract in `GOAL.md`. Record every
> explicit owner design change immediately, update and validate the referenced
> canonical sources, preserve all non-conflicting verified progress, and
> activate changes at the next safe receipt-backed boundary unless the owner
> explicitly orders immediate activation.

## Current objective

Continue the sequential specialist program from the runtime-selected active
specialist. Preserve completed specialists and passing checkpoints as immutable,
use the fixed 8,192-game baseline iteration and established research gates, keep
Kaggle submissions non-blocking, and transition to population round-robin only
after every required specialist is training-complete.

The Matchup Adapter V6 implementation is staged alongside the live V5 runtime.
V6 uses 64 fixed physical slots so ordinary archetype additions, retirements,
renames, and priority changes do not change tensor shapes or checkpoint format.
It activates only after compatibility receipts pass at a safe boundary.

The expanded strategic-head schema is implemented in the current
V5-compatible Dudunsparce learner and staged V6 architecture. It adds masked
action-value/factor/utility, tactical-outcome, opponent-response,
resource-forecast, game-phase, outcome-distribution, and remaining-turn
objectives. Every head with valid causal labels trains during full-model RL
updates and scheduled expert rehearsal; a head with no valid labels in a batch
is masked rather than assigned a false zero target. The checksum-bound learned
decision-fusion path is now the authoritative action path for the validated
Dudunsparce child and is mandatory in every generated successor bootstrap and
runtime. The immutable implementation-validation receipt is
`state/expanded_strategic_heads_validation_v1.json`; terminal serving
authorization additionally requires the exact runtime-child gate receipt.

Every successor specialist pre-stage must also research and materialize one
specialist-specific **current deck guide** objective modeled on the validated
Alakazam guide. The implementation uses one generic, checkpoint-declared guide
head and a checksum-bound per-specialist guide contract rather than adding
deck-named tensor code. Guide targets must be derived from cited public
strategy evidence and exact causally observable game state, remain auxiliary
training objectives, and never override the flat policy or consume hidden or
future information. Guide influence is temporary scaffolding: it ramps in
during bootstrap, remains strong only while separate evaluation evidence shows
positive deck/matchup win contribution, and decays toward zero as the learned
policy internalizes or surpasses it.

## Canonical sources

- Human workflow: `docs/RL_TRAINING_PROTOCOL.md`
- Numerical invariants: `config/rl_protocol.yaml`
- Mutable program state: `state/specialists.yaml`
- V6 slot and meta crosswalk registry: `state/matchup_adapter_roster.json`
- Runtime registry: `ops/specialist_runtime_registry_v1.json`
- Frozen registry: `ops/frozen_specialist_registry_v1.json`
- Transition graph: `ops/specialist_transition_graph.json`
- Compatibility projection for older controllers:
  `ops/current_goal_requirements.json`
- Live runtime truth: the canonical selector, managed-service state, and
  immutable receipts named by the compatibility projection

## Source precedence

When sources disagree, use this order:

1. The latest explicit owner decision recorded below.
2. Immutable receipts, the live selector, and managed-service state for facts
   about the currently executing runtime.
3. `config/rl_protocol.yaml` for numerical protocol invariants.
4. `state/specialists.yaml` and the matchup slot registry for mutable state.
5. Human documentation and dashboard projections.

Never allow stale prose or a dashboard label to override a live receipt. Never
silently weaken a gate, training isolation rule, frozen checkpoint guarantee,
or interactive-session safety rule.

## Design-change procedure

When the owner explicitly says that a requirement, design, canon, roster, gate,
or workflow has changed:

1. Record the decision in this ledger immediately and increment this file's
   revision.
2. Update the one typed canonical source that owns the changed value. Update
   derived documentation and dashboard projections, but do not create a second
   authority.
3. Validate the affected schemas and non-regression invariants.
4. Record the change as `staged` while an iteration is active.
5. Activate it at the next safe receipt-backed boundary. Immediate activation
   is allowed only when the owner explicitly requests it.
6. Record `activated_at_utc` and the activation receipt after the selector
   commits. A recorded decision remains canonical while activation is pending.

## Decision ledger

| Revision | Recorded at (UTC) | Decision | Activation |
|---|---|---|---|
| 1 | 2026-07-24 | Use this file as the adaptive goal gateway. Stage Matchup Adapter V6 with 64 fixed slots, stable slot identities, V5 compatibility, and registry-only ordinary roster edits. PTCGReplay is authoritative only for meta analysis and specialist priority. | Staged; activate V6 only at a safe receipt-backed boundary after compatibility validation. |
| 2 | 2026-07-24 | After the frozen Lucario specialist is registered, remove every external premium-holdout opponent whose canonical archetype is `lucario`; retain frozen Lucario as one additive `S+` opponent and preserve all historical results. With the current catalog this supersedes five of the eight external opponents, leaving three external plus four frozen specialists: seven opponents and 1,750 premium games. | Activated in the checksum-bound `frozen-specialists-r4` gate used by the live Dragapult/Dusknoir specialist. |
| 3 | 2026-07-24 | Reconcile the active specialist gate with the established 5/15 iteration canon and the authoritative LC50 numerical threshold. The historical Alakazam iteration-30 fallback remains immutable, but every current specialist uses LC50 beginning with its first transition-eligible evaluation at completed iteration 5. | Activated at the immutable Dragapult/Dusknoir iteration-4 commit and resumed at iteration 5 without replaying or discarding completed work. Receipt: `/home/inzi/poke-bot-agent/outputs/state/dragapult-dusknoir-lc50-iteration5-activation.json`; activated at `2026-07-24T22:15:52.700909+00:00`. |
| 4 | 2026-07-24 | Add the full expanded strategic-head system to V6: selected-action Q, factorized action-type/target/resource scorers, immediate action utility, 1/2/3-frame tactical outcomes, opponent response, next-own-decision resources, deterministic game phase, win/draw/loss distribution, and remaining turns. Keep every new head shadow-only initially, preserve the flat policy as authoritative, and mask every target that cannot be derived exactly and causally. | Staged for the next safe V6 cumulative-core/specialist handoff; the healthy live V5 specialist remains unchanged. |
| 5 | 2026-07-25 | Keep the goal gateway and planning documentation synchronized with the implemented 11-head V6 architecture, its exact cumulative 25-epoch schedule, target/schedule digests, checkpoint and rehearsal receipts, and dashboard projections. Implementation validation is not runtime activation. | Implementation validated by `state/expanded_strategic_heads_validation_v1.json`; activation remains staged for the next safe V6 cumulative-core/specialist handoff, with all runtime-enabled head lists empty and live V5 unchanged. |
| 6 | 2026-07-25 | After the first expanded-head cumulative core is frozen, permit separately registered compatibility derivatives of older frozen specialists to receive the V6 head architecture. Initialize every added head dormant and runtime-disabled. Preserve each original passing checkpoint byte-for-byte. A retrofit derivative is not a replacement passing checkpoint and may train only in a later explicitly scheduled retrofit phase while it is the sole active learner. | Materialized as five dormant, untrained derivatives under `_protected/retrofits/expanded-head-v1`, bound by `/home/inzi/poke-bot-agent/outputs/state/completed-specialist-expanded-head-retrofits-v1.json`. Every original source checksum is unchanged; runtime, gradients, serving, public-mix, gate, and Kaggle eligibility remain disabled. |
| 7 | 2026-07-25 | Train the complete expanded strategic-head set during the active specialist's ordinary full-model RL epochs and scheduled expert rehearsal whenever exact causal labels are present. Keep missing targets masked and keep every expanded head shadow-only for gameplay until separately activated. | Already active in the corrected Dudunsparce runtime without a restart. Selector `POKEBOT_EXPANDED_HEADS_ENABLED=1`; iteration-3 learner checkpoint `sha256:5e5ad03c2e7cbc396aee99f7adb08920ee2c38a992e14a751045e4f8aa3dd2d6` records nonzero labeled training for 10/11 heads, with `action_type` correctly fully masked for that batch. |
| 8 | 2026-07-25 | During preparation of every new specialist, research and build a specialist-specific current-deck guide objective as a training north star, following the validated Alakazam guide design. Use one generic head plus a checksum-bound deck-guide contract; cite the strategy sources, use only exact causal/public-state targets, mask unavailable targets, and keep the guide auxiliary so it cannot override the authoritative policy path. | Staged for the next specialist pre-stage and handoff. Do not interrupt the active Dudunsparce run; retrofit of already completed or active specialists requires a separate safe-boundary decision. |
| 9 | 2026-07-25 | Treat each current-deck guide as temporary learning scaffolding. Ramp its loss weight in during bootstrap, hold it only while receipt-backed deck/matchup evaluations show positive realized win contribution, and anneal it toward zero as the model internalizes or outperforms the guide. Estimate contribution only from training-ineligible guide-on/guide-off evaluation pairs; never adapt from training outcomes or leak evaluation games into updates. | Staged with revision 8 for the next specialist pre-stage and handoff. The active Dudunsparce lineage remains unchanged. |
| 10 | 2026-07-25 | Keep the research-control roster permanently fixed to exactly the four official agents. Never add a frozen specialist to research controls. Continue including every eligible frozen specialist in the separate premium/S+ holdout gate. | Active immediately as a contract clarification: the current fixed research-control registry already contains only the four official agents, while the premium holdout already adds all registered frozen specialists. No training restart or boundary migration is required. |
| 11 | 2026-07-25 | For every deck guide produced from strategy research, create a shareable subject-matter-expert write-up of no more than 10,000 words. Bind its path, checksum, word count, guide identity, and cited source set to the guide contract, and require it before successor pre-stage is ready. | Active for all future researched guides and added to the staged Marnie's Grimmsnarl ex guide without interrupting the live Dudunsparce lineage. |
| 12 | 2026-07-25 | Perform each specialist's deck-guide research and heuristic extraction in one dedicated highest-available-reasoning subagent whose scope contains exactly those two tasks. The production controller may validate and integrate the returned artifacts but must not expand that research agent into runtime, training, dashboard, or unrelated implementation work. | Active immediately for Marnie's Grimmsnarl ex and required for every later researched specialist guide. |
| 13 | 2026-07-25 | Treat both the dedicated two-task guide-research workflow and the owner's goal-path guide vision as protected non-regression requirements. The goal-path guidance ramps in, holds only while separate training-ineligible guide-on/guide-off evidence shows positive realized win contribution, and anneals toward zero after the policy internalizes or surpasses it. | Active in the canonical protocol and guarded by goal-gateway regression tests; no active training restart is required. |
| 14 | 2026-07-25 | Every expert-facing deck-guide write-up must primarily teach a human how to pilot that deck. It must cover setup, sequencing, first/second plans, resource and attack planning, bench and prize mapping, matchup plans, recovery lines, common mistakes, and decision checklists when evidence permits. Training-system and heuristic-audit material may appear only as a short appendix, never as the main guide. | Active for the staged Marnie's Grimmsnarl ex guide and required for every later researched specialist guide. |
| 15 | 2026-07-25 | Maximize completed iterations per wall-clock hour by scheduling every independent preparation, transfer, featurization, guide, pack, dashboard, and validation task concurrently across available hardware. Sequential execution is reserved only for true data dependencies, the single-active-specialist update invariant, immutable receipt barriers, and safe selector transitions. | Active immediately for orchestration and successor pre-stage; it does not restart or preempt the healthy active specialist. |
| 16 | 2026-07-25 | Make every causally available non-matchup, non-guide model head contribute to action selection for the active lineage and every successor specialist. This includes value, archetype, opponent-hand, opponent-remainder, lethal, prize-race, and all eleven expanded strategic heads. Matchup adapters remain causal-route gated and an absent current-deck guide remains an exact bypass. Replace the shadow-only serving contract with one checksum-bound learned decision-fusion path; train it jointly with the full model, preserve exact masking, and fail closed rather than silently dropping a required head. | Staged while the healthy Dudunsparce iteration is active. Implement and validate without preemption; activate at the next safe checkpoint boundary only after local/remote parity, deterministic inference, throughput, memory, and no-future-information receipts pass. This is mandatory for Dudunsparce after activation and every successor specialist. |
| 17 | 2026-07-25 | The all-head decision-fusion rule is a permanent successor-handoff invariant, not a one-specialist migration. Every generated successor bootstrap must enable and train the same 17-input fused action path automatically. A terminal serving-enabled child must receive its own exact 2,000-game premium and 1,000-game official evaluations; a flat-parent result can never freeze or authorize the child. Cross-platform parity requires float32 numerical agreement and identical greedy decisions rather than impossible byte identity across CPU kernels. | Active in the handoff generator and terminal runtime boundary. The successor contract regression requires all 17 inputs and `--decision-fusion`; the terminal handler now rejects flat-parent substitution and waits for the runtime child's checksum-bound two-gate receipt. |
| 18 | 2026-07-26 | Whenever a specialist becomes training-complete, automatically create exactly one one-shot Kaggle submission authorization bound to that specialist's immutable frozen passing checkpoint, upload bundle checksum, competition, and label. Submit or queue that single copy without blocking training. A quota or four-hour spacing wait remains pending; a pre-network authorization block is repaired oldest-first. This standing owner decision authorizes the missed Dudunsparce and Marnie's Grimmsnarl ex copies under the same exact identity checks. | Active immediately for the asynchronous submission queue and every future training-complete specialist. It does not interrupt the cumulative-core retry or active specialist training. |
| 19 | 2026-07-26 | Continue cumulative core distillation after every completed specialist, but never allow a rejected refresh candidate to stop specialist production. A failed refresh remains rejected and immutable; the next specialist immediately hot-starts from the latest checksum-accepted core, currently cumulative core v6. Future boundaries still attempt a new version from every frozen teacher. Every successor uses the validated 17-input causal decision-fusion path so all causally available non-matchup, non-guide heads participate in action selection; matchup adapters remain causal-route gated and an absent guide remains an exact bypass. | Active immediately at the failed post-Grimmsnarl v7 boundary. Start Garchomp from accepted core v6 through the managed handoff, while preserving the failed v7 attempt as diagnostic evidence and retaining per-specialist cumulative refreshes. |
| 20 | 2026-07-28 | Specialist transitions must be clean and automatic. Before a specialist may start, validate both its training launch and its exact terminal freeze/package/submission/handoff path, including its exact logical-ID 60-card representative. A normal terminal trainer exit directly starts the idempotent gate handler; the periodic supervisor remains recovery-only. Deterministic transition-input errors must fail before RL wall time is spent rather than strand an already passing specialist. | Active for Rocket's Mewtwo and every successor. Rocket's missing representative was repaired checksum-exactly, its passing checkpoint was frozen and accepted by Kaggle, Thwackey's aliased representative was pre-pinned, and the canonical trainer/handler units now enforce preflight plus `OnSuccess` chaining without interrupting the active cumulative-core handoff. |
| 21 | 2026-07-28 | Make the revision-20 transition guarantee transactional across immutable completion registration and the mutable priority projection. Selection may normalize only checksum-verified completed IDs and the exact outgoing active ID while the projection catches up; every other roster discrepancy remains fail-closed. Launch preflight must execute the exact specialist deck resolver and current-deck-guide registry/version dispatch used by training, including logical deck aliases and both supported historical checksum encodings. | Active for Thwackey and every successor. The post-Rocket v9 core passed its exact regression, Thwackey completed all 25 bootstrap epochs, the selector committed, exact deck/guide launch checks passed, and 8,192-game specialist RL started. Team Rocket's Spidops is registered as the next inactive guide target without changing the active Thwackey process. |
| 22 | 2026-07-28 | A successor with a missing causal matchup route must be repaired during one-ahead pre-stage, never after the outgoing specialist passes. Reuse checksum-verified historical public-prefix row shards by roster-name remapping, add independent causal calibration games for the missing route, and retain the established 0.93 precision and 10,000 weighted-support audit without weakening. Register only an inactive, immutable, checksum-bound router candidate; activation remains boundary-only. | Active for Team Rocket's Spidops and every analogous successor. The first v41 candidate was preserved as rejected at 93.12% precision but only 7,024 support. The expanded v42 candidate passed all 18 routes; Spidops passed at 93.15% precision and 15,703 support. The all-available-day expert scan found 630 exact causal decisions and therefore owns an explicit 630-decision pre-stage minimum rather than masquerading as the default 20,000. Thwackey production was not restarted or modified. |
| 23 | 2026-07-28 | The live specialist selector and its `POKEBOT_SPECIALIST_RUNTIME_ROOT` own one runtime registry for training preflight, terminal gate handling, and handoff validation. Stable/operator entry points must resolve that canonical registry rather than carrying an independently mutable copy. Every gate-handler CLI must bootstrap its own import path so a manual preflight and the managed service produce the identical command from any working directory. | Active without restarting Thwackey. The live and stable gate-handler entry points resolve the same Thwackey command, the handler self-bootstraps its runtime import path, and every successor prestage now validates the deterministic terminal handler before bootstrap exists. Hammer-Pult's terminal path is READY while its guide corpus builds; the focused transition/prestage suite passes 65 tests. Receipt: `/home/inzi/poke-bot-agent/outputs/state/specialist-transition-registry-convergence-v1.json`. |

## Non-regression invariants

- Never interrupt healthy active training merely to synchronize planning data.
- Run independent non-mutating preparation and validation work concurrently
  with healthy training and with each other. A controller must not serialize
  work merely for convenience; every sequential barrier must correspond to an
  explicit artifact, checksum, receipt, selector, or single-writer dependency.
- Use the available fleet concurrently for independent downloads,
  featurization, corpus assembly, guide preparation, CPU-pack construction,
  dashboard validation, and next-specialist readiness work, subject to the
  canonical memory guards and without starving the active training path.
- Never terminate or interfere with an interactive SSH, Codex, terminal,
  editor, Cursor, Grok, or Claude session.
- Only the active specialist may receive gradients or model updates.
- Completed specialists and original passing checkpoints remain immutable.
- Continue cumulative distillation after every completed specialist. A failed
  refresh candidate remains rejected and immutable but never blocks
  production: immediately hot-start the next specialist from the latest
  checksum-accepted core, then attempt another cumulative refresh at the next
  specialist boundary.
- Before starting any specialist, validate the same exact terminal
  freeze/package/submission/handoff command that will run after its gate.
  Successful terminal trainer exits directly invoke the idempotent gate
  handler; periodic supervision is recovery-only and may not be the normal
  transition mechanism.
- The launch preflight must execute the same exact 60-card specialist resolver
  and current-deck-guide module/version dispatch as the trainer. A guide
  contract or representative file existing on disk is insufficient if the
  active runtime cannot resolve it.
- A missing successor matchup route is a one-ahead pre-stage task. Its
  candidate must remain inactive and must pass the established causal-router
  precision and support audit before selection; never weaken either threshold
  to make a transition ready.
- At a freeze boundary, a mutable priority projection may lag immutable frozen
  registration only for the exact checksum-verified completed IDs and outgoing
  active ID supplied by the handoff. Normalize those IDs transactionally;
  never tolerate unrelated missing, duplicated, reordered, or unknown roster
  entries.
- Every training-complete specialist creates exactly one checksum-bound,
  single-use Kaggle submission authorization automatically. Submission remains
  asynchronous and non-blocking, and delayed copies always retain the original
  frozen passing checkpoint identity.
- Expanded-head retrofits use new derivative identities and never rewrite,
  replace, re-freeze, or silently promote an original passing checkpoint.
- Retrofitted heads remain dormant and runtime-disabled until their own
  training, validation, checksum registration, and activation receipt exist.
- The active specialist's expanded strategic heads receive gradients in every
  full-model RL or rehearsal update for which their exact causal labels are
  present; missing labels remain masked.
- Each newly prepared specialist has exactly one checksum-bound current-deck
  guide contract and generic guide head. Guide evidence is cited, labels are
  causal and specialist-specific, missing labels are masked, and no guide may
  alter the authoritative action path without a separate activation receipt.
- Each researched current-deck guide has a source-cited, checksum-bound expert
  write-up of at most 10,000 words. The write-up is required for pre-stage
  readiness and must be suitable for review by world-champion subject-matter
  experts.
- The expert write-up is always a practical human pilot guide first.
  Training-system or heuristic-audit details are permitted only in a short
  appendix and may never replace the how-to-play content.
- Guide research and heuristic extraction run in a dedicated
  highest-available-reasoning subagent with exactly those two responsibilities.
  Its artifacts are validated by the controller before integration.
- Guide-weight changes require a checksum-bound schedule and separate
  training-ineligible guide-on/guide-off evaluation evidence. Training wins,
  replay outcomes, and formal gate games never directly tune guide weight.
- The learned decision-fusion path must consume every causally available
  non-matchup, non-guide head. Matchup adapters remain causal-route gated and
  an absent guide remains an exact bypass. Until the revision-16 activation
  receipt exists, the current iteration retains its already-pinned flat-policy
  action path; after activation, silently omitting a required head fails
  closed.
- Research evaluations never enter training or replay data.
- Research controls contain exactly the four official agents and never frozen
  specialists. Eligible frozen specialists remain required inference-only
  opponents in the separate premium/S+ holdout gate.
- Runtime identity comes from one selector, never from service or directory
  names.
- V6 permits at most one matchup slot per decision; unknown, retired,
  ambiguous, or insufficiently observed matchups exactly bypass the bank.
- V6 slot identities never move and retired slots are never automatically
  reused.

## Safe resume

Read this file first, then read the referenced machine protocol, mutable state,
slot registry, runtime selector, and latest immutable receipts. Continue the
healthy managed workload if their evidence agrees. If a staged change is
pending, finish its validation in parallel and apply it only at its declared
safe boundary.
