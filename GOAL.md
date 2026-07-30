# Pokémon RL Goal Gateway

Schema: `poke_bot.goal_gateway/v1`  
Revision: `67`

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
Kaggle submissions non-blocking, and transition immediately after the final
required Slowking specialist to a separately versioned, up-to-date Alakazam
refresh in the validated final-submission format. Prefer a checksum-bound
compatibility hot start from the immutable existing Alakazam checkpoint when it
can pass the required migration and parity receipts; never rewrite that parent.
Continue to the Marnie's Grimmsnarl ex refresh only after the new Alakazam
independently passes, freezes, and registers. Transition to population
round-robin only after both refresh versions independently pass, freeze, and
register.

Matchup Router Format 6 is active in the Teal Mask Ogerpon ex lineage. It uses 64 fixed
physical slots so ordinary archetype additions, retirements, renames, and
priority changes do not change tensor shapes or checkpoint format. Its
compatibility receipts passed at the Teal receipt-backed activation boundary;
older Router Format 5 checkpoints remain immutable historical lineage.
This router-format number is independent of the training-core revision and
accepted-policy generation. The latest checksum-accepted production policy is
Accepted Policy Generation 9
(`sha256:7d9b60e68f4c51bb931298ae3941e5b7bddf1370566b23d18acadd33e8357056`);
the post-Thwackey Policy Generation 10 candidate was attempted and immutably
rejected by the unchanged gameplay-regression gate. The post-Spidops
Policy Generation 11 attempt was then
rejected during pretraining validation before it created a candidate or ran
gameplay regression. The post-Hammer-Pult Policy Generation 12 attempt was also
rejected during pretraining validation. Teal Mask Ogerpon ex therefore
hot-started from Accepted Policy Generation 9; later successors continue from
that accepted policy until a later cumulative candidate passes.

User-facing status must distinguish three independently versioned surfaces:
**Training Core Revision 10** is the current training/control implementation,
**Matchup Router Format 6** is the active matchup-router checkpoint layout, and
**Accepted Policy Generation 9** is the latest accepted production policy
checkpoint. Policy attempts 10, 11, and 12 remain rejected. Bare `Vn` labels
are not permitted for any of these current-status surfaces.

The expanded strategic-head schema is implemented in the current
Model Format 5-compatible Dudunsparce learner and active Model Format 6
successor architecture. It adds masked
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
Alakazam guide. The implementation uses one generic training-only curriculum
contract plus a checksum-bound per-specialist guide contract rather than adding
deck-named runtime tensor code. Guide signals must be derived from cited public
strategy evidence and exact causally observable game state, remain auxiliary
training metadata, and never override the policy, supply action logits, or
consume hidden or future information. Beginning with Archaludon ex, the guide
steers only observed-target learning on the independently fused learned heads;
it never becomes an imitation target. Guide influence is temporary
scaffolding: it ramps in during bootstrap, remains strong only while separate
evaluation evidence shows positive deck/matchup win contribution, and decays
toward zero as the learned policy internalizes or surpasses it.

## Canonical sources

- Human workflow: `docs/RL_TRAINING_PROTOCOL.md`
- Numerical invariants: `config/rl_protocol.yaml`
- Mutable program state: `state/specialists.yaml`
- Router Format 6 slot and meta crosswalk registry: `state/matchup_adapter_roster.json`
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
| 4 | 2026-07-24 | Add the full expanded strategic-head system to V6: selected-action Q, factorized action-type/target/resource scorers, immediate action utility, 1/2/3-frame tactical outcomes, opponent response, next-own-decision resources, deterministic game phase, win/draw/loss distribution, and remaining turns. Keep every new head shadow-only initially, preserve the flat policy as authoritative, and mask every target that cannot be derived exactly and causally. | Staged for the next safe cumulative-core/specialist handoff carrying the V6 architecture; the healthy live V5 specialist remains unchanged. |
| 5 | 2026-07-25 | Keep the goal gateway and planning documentation synchronized with the implemented 11-head V6 architecture, its exact cumulative 25-epoch schedule, target/schedule digests, checkpoint and rehearsal receipts, and dashboard projections. Implementation validation is not runtime activation. | Implementation validated by `state/expanded_strategic_heads_validation_v1.json`; activation remains staged for the next safe cumulative-core/specialist handoff carrying the V6 architecture, with all runtime-enabled head lists empty and live V5 unchanged. |
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
| 19 | 2026-07-26 | Continue cumulative core distillation after every completed specialist, but never allow a rejected refresh candidate to stop specialist production. A failed refresh remains rejected and immutable; the next specialist immediately hot-starts from the latest checksum-accepted core, then the next cumulative-core version is attempted at the following boundary. Future boundaries still attempt a new version from every frozen teacher. Every successor uses the validated 17-input causal decision-fusion path so all causally available non-matchup, non-guide heads participate in action selection; matchup adapters remain causal-route gated and an absent guide remains an exact bypass. | Historical boundary outcome: Garchomp started from accepted Core Generation 6 after the failed post-Grimmsnarl Core Generation 7 attempt. The current accepted core is Core Generation 9; Core Generations 10 and 11 were attempted and rejected at their recorded validation stages. |
| 20 | 2026-07-28 | Specialist transitions must be clean and automatic. Before a specialist may start, validate both its training launch and its exact terminal freeze/package/submission/handoff path, including its exact logical-ID 60-card representative. A normal terminal trainer exit directly starts the idempotent gate handler; the periodic supervisor remains recovery-only. Deterministic transition-input errors must fail before RL wall time is spent rather than strand an already passing specialist. | Active for Rocket's Mewtwo and every successor. Rocket's missing representative was repaired checksum-exactly, its passing checkpoint was frozen and accepted by Kaggle, Thwackey's aliased representative was pre-pinned, and the canonical trainer/handler units now enforce preflight plus `OnSuccess` chaining without interrupting the active cumulative-core handoff. |
| 21 | 2026-07-28 | Make the revision-20 transition guarantee transactional across immutable completion registration and the mutable priority projection. Selection may normalize only checksum-verified completed IDs and the exact outgoing active ID while the projection catches up; every other roster discrepancy remains fail-closed. Launch preflight must execute the exact specialist deck resolver and current-deck-guide registry/version dispatch used by training, including logical deck aliases and both supported historical checksum encodings. | Active for Thwackey and every successor. The post-Rocket v9 core passed its exact regression, Thwackey completed all 25 bootstrap epochs, the selector committed, exact deck/guide launch checks passed, and 8,192-game specialist RL started. Team Rocket's Spidops is registered as the next inactive guide target without changing the active Thwackey process. |
| 22 | 2026-07-28 | A successor with a missing causal matchup route must be repaired during one-ahead pre-stage, never after the outgoing specialist passes. Reuse checksum-verified historical public-prefix row shards by roster-name remapping, add independent causal calibration games for the missing route, and retain the established 0.93 precision and 10,000 weighted-support audit without weakening. Register only an inactive, immutable, checksum-bound router candidate; activation remains boundary-only. | Active for Team Rocket's Spidops and every analogous successor. The first v41 candidate was preserved as rejected at 93.12% precision but only 7,024 support. The expanded v42 candidate passed all 18 routes; Spidops passed at 93.15% precision and 15,703 support. The all-available-day expert scan found 630 exact causal decisions and therefore owns an explicit 630-decision pre-stage minimum rather than masquerading as the default 20,000. Thwackey production was not restarted or modified. |
| 23 | 2026-07-28 | The live specialist selector and its `POKEBOT_SPECIALIST_RUNTIME_ROOT` own one runtime registry for training preflight, terminal gate handling, and handoff validation. Stable/operator entry points must resolve that canonical registry rather than carrying an independently mutable copy. Every gate-handler CLI must bootstrap its own import path so a manual preflight and the managed service produce the identical command from any working directory. | Active without restarting Thwackey. The live and stable gate-handler entry points resolve the same Thwackey command, the handler self-bootstraps its runtime import path, and every successor prestage now validates the deterministic terminal handler before bootstrap exists. The cycle contract reads its mutable specialist projection from `/home/inzi/poke-bot-agent/state/specialists.yaml`, eliminating the stale deployment copy that had incorrectly reselected Spidops. Hammer-Pult's terminal path is READY while its guide corpus builds; the focused transition/prestage suite passes 65 tests. Receipt: `/home/inzi/poke-bot-agent/outputs/state/specialist-transition-registry-convergence-v1.json`. |
| 24 | 2026-07-28 | Team Rocket's Spidops is the mandatory and sole successor after Thwackey. Missing Spidops inputs must block lower-priority selection and trigger recovery from the full public replay history, including newly published days, rather than falling through to Hammer-Pult or accepting the 12-game/630-decision historical shard as the intended corpus. The current public rolling window is June 26 through July 27 and must yield at least 16,639 exact acting-seat Spidops games before pre-stage is ready. Public deck-to-archetype identities must be checksum-bound and used to repair the local classifier's false cross-archetype precedence; causal replay conversion, training/evaluation isolation, guide binding, and all existing gates remain unchanged. | Active immediately for inactive successor selection and corpus preparation. Thwackey remains the healthy non-preemptible active learner. Hammer-Pult preparation may continue independently but it cannot be selected or activated before Spidops without a later explicit owner decision. Spidops model activation remains boundary-only after Thwackey completes. |
| 25 | 2026-07-28 | Deprioritize `dragapult-blaziken` and `dragapult-dudunsparce`. Both remain required specialists, but they sort behind every other unfinished specialist; their relative order remains Dragapult/Blaziken before Dragapult/Dudunsparce. | Staged while Spidops iteration 1 is active. Do not change or restart Spidops, and do not replace the already prepared Hammer-Pult successor. Activate the revised ordering at the next receipt-backed successor-selection boundary. |
| 26 | 2026-07-28 | Treat Hammer-Pult as a nonlinear deck. Its sparse causal guide is auxiliary scaffolding, not a sufficient strategy system by itself. Before Hammer-Pult may retain pre-stage-ready status or bootstrap, checksum-bind and validate the fused-policy systems for branching setup and pivot choice, attacker/engine preservation, typed-Energy and resource planning, Phantom Dive target and prize routing, disruption under stochastic Hammer outcomes, bench/recovery routing, and guide-weight annealing. Unsupported exact guide labels remain masked; hidden or future information remains prohibited. | Staged while Spidops iteration 1 is active. Revalidate and reissue Hammer-Pult's pre-stage receipt before its bootstrap; fail closed if any nonlinear system or binding is absent. Do not change or restart Spidops. |
| 27 | 2026-07-28 | Add Teal Mask Ogerpon ex as a new required specialist immediately after Hammer-Pult. It requires its own exact archetype identity, 60-card representative, full available public corpus, causal matchup route, current-deck guide, nonlinear/specialist systems where declared, exact 25-epoch bootstrap, fused policy, gates, freeze, and one-shot Kaggle authorization. | Staged during Spidops iteration 1. Dedicated current-guide research and causal heuristic extraction may proceed in parallel, but training and activation remain boundary-only after Hammer-Pult. |
| 28 | 2026-07-28 | Remove `dragapult-blaziken` and `dragapult-dudunsparce` from the required specialist plan entirely. Do not select, bootstrap, gate, freeze, submit, or count either deck toward program completion. Preserve any historical corpus, router, and audit artifacts as non-planning evidence; stable matchup slots are not deleted or reused merely because the specialist target is removed. This supersedes revision 25. | Active immediately in planning and staged for the next receipt-backed selector projection. The active Spidops run and prepared Hammer-Pult assets remain unchanged. |
| 29 | 2026-07-28 | Fix the post-Spidops successor prefix to `hammer-pult`, then `teal-mask-ogerpon-ex`, then `archaludon-ex`. Other retained unfinished specialists follow only after Archaludon ex. Missing inputs for any strict-prefix target block fallback selection and trigger public-data recovery and pre-stage completion. | Staged while Spidops iteration 1 is active; activate at each normal receipt-backed handoff without preempting the current learner. |
| 30 | 2026-07-29 | Keep Core Generation numbers and Router Format checkpoint-compatibility numbers in separate namespaces everywhere user-facing plans, protocol state, and dashboards display them. The current cumulative-core state is accepted Core Generation 9, rejected Core Generation 10 gameplay regression, and rejected Core Generation 11 pretraining validation. “Router Format 6” must never be presented as the current cumulative-core generation. | Active immediately as a planning and display clarification. Runtime, selectors, checkpoints, gates, and the healthy Hammer-Pult learner are unchanged. |
| 31 | 2026-07-29 | When Teal Mask Ogerpon ex becomes training-complete, create two checksum-bound Kaggle submissions from the same immutable frozen checkpoint and exact 60-card deck. Copy 1 must choose to go first whenever the competition exposes the legal turn-order choice; copy 2 must choose to go second whenever that choice is exposed. Each bundle requires its own digest-bound turn-order attestation, queue identity, label, authorization use, and submission receipt. If the environment does not expose a turn-order choice, both agents continue normally without inventing one. The second copy remains asynchronous and non-blocking under the unchanged daily quota and four-hour spacing rules. | Staged immediately for Teal Mask packaging, terminal preflight, and the future submission queue. It does not modify, restart, or preempt Hammer-Pult. The default for every other specialist remains one first-preferring submission unless a later owner decision says otherwise. |
| 32 | 2026-07-29 | The required `teal-mask-ogerpon-ex` specialist and guide are the exact current public archetype-151 deck shown by the owner: 3 Mega Kangaskhan ex, 3 Meowth ex, 3 Teal Mask Ogerpon ex, 2 Latias ex, 2 Raging Bolt ex, one each of Fezandipiti ex, Lillie's Clefairy ex, Passimian, and Wellspring Mask Ogerpon ex, with the displayed Area Zero/Crispin/Energy Switch/Glass Trumpet multi-Energy engine. Record the local deck name `Slop Box` as its primary human-facing alias and `Raging Bolt Ogerpon` as a searchable competitive-family alias while retaining the stable logical program ID. The older Japan Championships Ogerpon Box list is not this specialist's guide or representative. | Active immediately for inactive Teal guide identity, representative binding, corpus checks, and pre-stage validation. Any receipt bound to the older guide identity fails closed and must be reissued. Hammer-Pult training remains unchanged. |
| 33 | 2026-07-29 | Create exactly one additional Kaggle submission for `hops-trevenant` from its immutable owner-accepted iteration-10 checkpoint and exact checksum-bound 60-card representative. This one-off copy must choose to go second whenever the competition exposes the legal turn-order choice; if no choice is exposed, it continues normally without inventing one. Bind the upload to its own second-preferring bundle digest, turn-order attestation, single-use authorization, label, and submission receipt. This does not replace or rewrite either historical Hops submission and does not change the first-preferring default for other specialists. | Activated at `2026-07-29T17:27:30.383000Z`. Kaggle submission `55088551` completed with public score `600.0` from bundle `sha256:0f6b10a2198a5aa368220d0d34674b020bec9fffeb1fcf9c85d529cd8d0828b5`; the exact one-shot authorization was consumed before the successful upload invocation. The submission did not change or preempt Hammer-Pult. Its later independent iteration-15 recovery found one missing static deck-mix asset; the checksum-exact canonical file was restored and the existing production supervisor resumed the same selector and learner. |
| 34 | 2026-07-29 | Remove `walrein` from the required specialist program. Do not select, bootstrap, train, gate, freeze, submit, or count Walrein toward completion. Preserve its historical corpus, classifier, router, adapter-slot, and audit evidence without deleting, reindexing, or reusing its stable matchup identity. Separately prepare the owner's exact 60-card Slowking list and a source-cited causal deck guide as research-only assets. Slowking does not replace Walrein in the order, is not a required specialist or matchup route, and must not be added to the selector, runtime registry, pre-stage state, services, dashboard training queue, or completion count without a later explicit owner decision. | Active immediately in planning and staged for the next receipt-backed selector projection. Slowking research artifacts may be built and validated now with zero training authority. The healthy Hammer-Pult learner, strict Teal Mask Ogerpon ex and Archaludon ex prefix, current matchup bank, and active selector remain unchanged. |
| 35 | 2026-07-29 | Treat `Slop Box` as the primary local name for the exact Teal Mask Ogerpon ex plus Mega Kangaskhan ex deck shell. Mega Kangaskhan ex is part of the deck's defining identity; Raging Bolt ex is an included attacker and remains only a secondary search/taxonomy alias, not the primary deck label. | Active immediately for the inactive Teal guide, dashboard naming, and future receipts. The exact 60-card representative and stable logical ID remain unchanged, and Hammer-Pult remains untouched. |
| 36 | 2026-07-29 | After all 16 current required specialists are training-complete and frozen, and before population training unfreezes, train new separately versioned refreshes in strict order: `alakazam`, then `marnie-s-grimmsnarl-ex`. At each refresh start, resolve the newest checksum-accepted cumulative core and the then-current canonical training structures rather than pinning today's core generation or schema digests. Attempt the normal cumulative-core refresh at each predecessor boundary; preserve a rejected candidate immutably and use the newest accepted fallback. Each refresh must use the current exact 25-epoch bootstrap, latest-20 specialist corpus and current-deck guide contract, causal learned decision fusion, every then-canonical causally available strategic head, matchup architecture, 8,192-game RL iteration, 5-RL/5-rehearsal cadence, and unchanged gates. Preserve the original passing Alakazam and Grimmsnarl checkpoints and all historical evidence byte-for-byte; the old V5-bound dormant Alakazam compatibility derivative is not the requested new version. | Staged for a dedicated post-fleet phase. It does not add either refresh to the 16-specialist roster or today's selector, runtime registry, pre-stage, service, or training queue; Hammer-Pult and its strict successors remain unchanged. At the fleet-complete receipt boundary, population transition fails closed until the new Alakazam refresh, its following core boundary, and the new Grimmsnarl refresh each independently complete, freeze, and register. |
| 37 | 2026-07-29 | If Hammer-Pult's terminal iteration-15 performance gate does not pass, do not extend, retry, or restart Hammer-Pult. Preserve the complete failed-gate evidence, freeze and register the latest exact immutable iteration-15 checkpoint as `ceiling_accepted` without calling it a measured pass, create its normal checksum-bound Kaggle submission or queue it under the existing quota rules, and continue strictly to Teal Mask Ogerpon ex. This authorization applies to a completed auditable performance-gate failure; a missing, corrupt, mismatched, or incomplete checkpoint or receipt remains fail-closed and may not be submitted. | Active at the current Hammer-Pult terminal boundary. The running managed handler already uses `--accept-ceiling-and-continue` and the canonical `freeze_submit_and_continue_without_false_pass` contract, so no service restart or runtime mutation is required. Teal activation still waits for its exact full-history corpus, pre-stage, and normal receipt-backed handoff. |
| 38 | 2026-07-29 | Compute the unchanged four-hour Kaggle queue spacing limit from the second-most-recent logical submission, not the most recent submission. Reconciled Kaggle and local queue rows for the same upload count once, keyed by submission id or checkpoint-bound label. With fewer than two prior logical submissions there is no spacing anchor. Daily quota, checksum identity, one-shot authorization, oldest-first ordering, and non-blocking training behavior remain unchanged. | Active immediately in the asynchronous queue processor. Re-evaluate the pending Hammer-Pult copy under the corrected anchor and submit it now if its identity, authorization, and daily quota checks pass. |
| 39 | 2026-07-29 | Replace ambiguous bare `Vn` labels in current plans, protocol state, and dashboards with two explicit namespaces: `Core Generation N` for cumulative policy/core lineage and `Router Format N` for matchup-router checkpoint compatibility. The live boundary activates Router Format 6. Runtime evidence remains authoritative for the core lineage: Core Generation 9 is the latest accepted checkpoint, while Core Generations 10, 11, and 12 are immutable rejected attempts. | Activated at the Teal receipt-backed boundary. Router activation receipt: `/home/inzi/poke-bot-agent/outputs/state/matchup-v6-fleet-activation-v3.json` (`sha256:6d4c314483beebf824379e5f67cb597176c67a52e8e576feebdf08b1941bd75c`). This naming clarification does not relabel a rejected core as accepted or alter gates. |
| 40 | 2026-07-29 | Use three explicit user-facing namespaces instead of calling every surface `Vn`: `Core System Revision 10` for the current training/control implementation, `Router Format 6` for matchup-router checkpoint compatibility, and `Policy Checkpoint Generation N` for cumulative policy checkpoints. The accepted production checkpoint remains Policy Checkpoint Generation 9; attempts 10, 11, and 12 remain rejected. | Active immediately for the canonical plan and dashboard. This is a display/terminology change only and does not mutate runtime, selectors, checkpoints, or gates. |
| 41 | 2026-07-29 | Shorten and clarify the current-status names everywhere user-facing: `Training Core Revision 10`, `Matchup Router Format 6`, and `Accepted Policy Generation N`. Do not derive or display bare `Vn` labels from checkpoint filenames. The three counters remain independent: the training core is revision 10, the router checkpoint layout is format 6, and the latest accepted policy is generation 9; rejected policy attempts 10, 11, and 12 retain their evidence. | Active immediately for canonical projections and the dashboard. Machine schema identifiers, receipt paths, and immutable historical artifact names keep their exact version suffixes. Runtime, selectors, checkpoints, and gates are unchanged. |
| 42 | 2026-07-30 | Implement current-deck guide importance as the owner's realized-win curve: ramp rapidly during early learning, hold a materially useful plateau while guide contribution remains positive, then decay toward zero as the policy internalizes the guide or its separate contribution becomes nonpositive. For the active Slop Box lineage, the exact iteration-4 diagnosis found a 92.8% empty-bench decline rate in a bounded causal setup audit while the guide weight was only `0.05`; this authorizes one corrective clean-boundary increase to `0.25`. The guide remains auxiliary and may not override runtime actions. Later hold/decay decisions still require training-ineligible guide-on/guide-off evidence. | Activated at `2026-07-30T04:17:16.732246Z` after immutable iteration-5 commit `sha256:db28b6681eee5f989a3a8f98e6f7299fbdce2135c05fc1df1b23a8bba61283e4`. Iteration 6 started under weight `0.25` with a new managed PID and preserved selector identity. Receipt: `/home/inzi/poke-bot-agent-deployments/dudunsparce-decision-fusion-v1/outputs/pure_rl/pure_rl_teal-mask-ogerpon-ex_temporal1_8k_v1_20260723/design_migrations/migration_0006.json`. |
| 43 | 2026-07-30 | Apply the revision-42 realized-win importance curve to every current-deck guide, not only Slop Box. After the fixed bootstrap ramp, each guide may increase through the canonical fleet ramp while its separate training-ineligible guide-on/guide-off realized-win lower confidence bound is positive, hold at its evidence-supported plateau, and move through the canonical decay schedule after two consecutive nonpositive reviews. Fleet controllers are authorized to adjust weights within the `0.00`–`0.50` auxiliary range at clean receipt-backed boundaries without a new per-deck owner decision. Guide weights remain auxiliary, never directly select actions, and training, replay, and formal-gate outcomes remain ineligible to tune them. | Active immediately for the generic schedule contract and every inactive/future guide. For the active Slop Box lineage, preserve revision 42's exact staged `0.05`→`0.25` migration after the immutable iteration-5 commit; do not broaden that pending boundary transaction or interrupt its healthy trainer. |
| 44 | 2026-07-30 | Scope revision 43 prospectively to future specialist training runs only. Do not retrofit, reevaluate, retrain, rewrite guide weights, or attach the new adaptive-review lifecycle to completed, frozen, or already-started specialist runs. Preserve their checkpoints, guide weights, and receipts byte-for-byte. The already-activated Slop Box `0.05`→`0.25` revision-42 correction remains its explicit one-off exception and does not authorize further revision-43 adaptation during that run. | Active immediately for planning and future specialist registration. Install and bind the generic adaptive policy when the next specialist runtime is created, beginning with Archaludon ex and all later newly started specialists and separately versioned post-fleet refreshes. Do not restart or mutate active Teal training and do not alter any historical specialist. |
| 45 | 2026-07-30 | Make the guide-weight curve an actual learning-loss curve, not a label, dashboard-only score, or runtime action bias. For every future run covered by revision 44, the active receipt-backed weight multiplies the masked, confidence-weighted current-deck guide cross-entropy inside each eligible training update. Evidence reviews may move that multiplier up the canonical ramp, hold it at the evidence-supported summit, or move it down the canonical decay, but the guide never directly selects an action. Because the 1,000-pair shadow evaluation is nonblocking, a completed review establishes the earliest eligible next iteration; application occurs at the first later available five-iteration hard pause and the application receipt records the actual boundary. | Active immediately in the future specialist policy and typed protocol, beginning with Archaludon ex. This clarification does not alter active Teal or any completed, frozen, or already-started run. |
| 46 | 2026-07-30 | Interpret the “up the mountain” guide curve as actual supervised learning pressure. On every eligible future-specialist update, the active receipt-backed guide weight scales the guide loss before backpropagation, so ascending the ramp increases the guide gradient contribution to shared policy learning, the summit holds that contribution only while realized-win evidence remains positive, and descending the decay reduces it toward zero. The curve may not be implemented as elapsed-time-only progression, guide-head-only bookkeeping, a dashboard score, or serving-time action bias. | Active immediately as a non-regression clarification of revision 45, beginning with Archaludon ex. It changes no active, completed, frozen, or already-started run and requires no service restart. |
| 47 | 2026-07-30 | Repair the Teal iteration-7 cgroup OOM without sacrificing its immutable iteration-6 commit or manually restarting the recovered learner. Replace the unsafe 128-worker/128-worker-minimum selector with a 48-worker next-start ceiling, a 32-worker rebalance floor, and a 12 GiB free-RAM floor. Lower the managed unit's soft `MemoryHigh` guard from 110 GiB to 100 GiB while retaining the 116 GiB hard cap. | Fully activated by managed recovery. Systemd first recovered the OOM-interrupted noncommitted attempt under PID `2307796` without manual intervention. That inherited 128-worker process later failed closed after leaf timeouts left an exact-collection shortfall of `8146/8192`; its uncommitted iteration-8 shard was quarantined by receipt. Systemd then automatically restarted under PID `2864742`, activating the 48-worker selector, 32-worker rebalance floor, 12 GiB free-RAM floor, and 100 GiB soft guard while preserving immutable iteration 7. |
| 48 | 2026-07-30 | The Slop Box guide did not produce adequate game wins: iteration 11 reached high action-imitation accuracy but failed promotion, remained far below the strong-game gate, and regressed the isolated official controls. Lower the active Teal current-deck guide loss weight; further imitation pressure should be reduced for this run. Preserve every completed iteration and evaluation receipt. This is a second explicit Teal-only owner override and does not make revision 43 retroactive: Archaludon ex and later future runs retain their separate training-ineligible guide-on/guide-off realized-win controller. | Superseded before activation by revision 49's explicit nonzero target. No runtime change occurred under the initially inferred zero-weight staging. |
| 49 | 2026-07-30 | Clarify revision 48: do not reduce the Slop Box guide weight to zero. Move it from `0.25` to the next bounded decay step, `0.15`, retaining reduced auxiliary guide learning while substantially lowering imitation pressure. | Superseded before activation by revision 50's exact `0.05` target. No runtime change occurred under the inferred `0.15` staging. |
| 50 | 2026-07-30 | Set the revision-48 Slop Box guide-weight reduction target to exactly `0.05`, superseding revision 49's inferred `0.15`. Preserve a small auxiliary guide signal while returning imitation pressure to the original low weight. | Activated at `2026-07-30T14:03:39.700214Z` after immutable iteration-12 commit `sha256:d9627b0884d136768ded779042f84ccb1331369e78a97ed643fa4d9ce28dda13`. Iteration 13 started under exact weight `0.05` with managed PID `3848706`, preserved selector identity, unchanged trainer/train-module checksums, and restored stop protection. Receipt: `/home/inzi/poke-bot-agent/outputs/pure_rl/pure_rl_teal-mask-ogerpon-ex_temporal1_8k_v1_20260723/design_migrations/migration_0008.json` (`sha256:217e9f73c151df6646a7005a652547d7fa9e4f147392f62eba614db4b2e9621d`). |
| 51 | 2026-07-30 | For future specialist runs only, make the current-deck guide a bounded curriculum for the causal strategic heads rather than a policy-imitation target. A guide-labeled row may focus training on the relevant outcome-backed heads, but it may not provide runtime inputs, directly select an action, replace observed outcome targets, or make the policy depend on imitation. Keep the guide multiplier as a small, receipt-backed learning-pressure control on this curriculum. | Active prospectively beginning with Archaludon ex. Do not retrofit this architecture into the already-started Slop Box run or any completed/frozen specialist. The future implementation and pre-stage validation must prove that guide supervision terminates at training-only strategic-head objectives and that the fused policy remains win/outcome trained. |
| 52 | 2026-07-30 | In the active Slop Box run, rebalance the non-guide auxiliary losses at the next clean boundary. The iteration-10 full expert rehearsal showed `tactical_outcome` raw train loss `3.977706`, so its `0.05` multiplier contributed about `0.198885` weighted loss by itself, versus about `0.032969` from the then-`0.25` guide and roughly `0.03`–`0.05` from each other material strategic head. Lower only the Teal `tactical_outcome` multiplier to `0.01`, which would have contributed about `0.039777` on the same measured batch. Preserve all other head weights, the exact `0.05` guide weight, completed iterations, runtime fusion, and selector identity. | Activated at the clean iteration-13/14 boundary. Iteration 14 resumed under guide weight `0.05` and tactical-outcome weight `0.01`; the migration changed only `learner.expanded_head_loss_weight_overrides` plus the source digest, preserved the verified iteration-13 checkpoint and selector identity, and restored `RefuseManualStop=yes`. Receipt: `/home/inzi/poke-bot-agent/outputs/pure_rl/pure_rl_teal-mask-ogerpon-ex_temporal1_8k_v1_20260723/design_migrations/migration_0009.json`. |
| 53 | 2026-07-30 | Future specialist guide curricula may train both independent strategic branches whose typed outputs enter the existing fusion path and fully unfused shadow heads of the kind specified in `docs/FINAL_MODEL_CAPACITY_AND_DECISION_REPLAY_PLAN.md`. Every head must declare `fusion_role: fused_input` or `fusion_role: shadow_unfused`. A shadow-unfused head may shape its own branch and the shared representation during training and may provide diagnostics/ablations, but it has no direct action-logit, action-selection, runtime-input, or serving authority. Moving a shadow head into fusion is a separate architecture change with step-zero parity and gameplay receipts. | Active prospectively beginning with Archaludon ex guide-curriculum pre-stage and retained for later H10-I research. Does not retrofit Slop Box or historical specialists and does not authorize pre-fleet H10 computation. |
| 54 | 2026-07-30 | Correct future setup and bench-development errors with a dedicated option-conditioned `setup_board_outcome` branch, not another guide-imitation policy head. The branch is a fully unfused `d_model → 512 → 9` shadow MLP. On selected, complete setup-active/setup-bench stages only, its observed targets are the six already-causal next-own-decision resource-forecast values plus terminal win/draw/loss; unavailable, truncated, unselected, or malformed targets are masked. Balance setup-active and setup-bench reductions so rare bench rows are not diluted by main-phase decisions. The ordinary branch weight is `0.025`; the receipt-backed guide weight may add `guide_confidence × observed-target loss`, but the guide target index may never enter this loss. | Active prospectively beginning with Archaludon ex. `fusion_role: shadow_unfused`; no action-logit, fusion, runtime-input, action-selection, or serving authority. Slop Box and historical runs remain byte-for-byte unchanged. Archaludon bootstrap stays fail-closed until a checksum-bound validation receipt proves target parity, guide-target permutation invariance, nonzero branch/shared-representation gradients, zero policy-head/fusion gradients, shadow ablation action parity, coverage/calibration, and exact role-map binding. |
| 55 | 2026-07-30 | Supersede revisions 53 and 54 only where they isolated a future strategic head from action choice. At any decision, the action must be based on the causal board state and legal options through multiple independently computed head views; every future strategic head must have an explicit bounded route into the learned decision-fusion action score. “Independent” or “unfused” describes how a head computes its typed view before fusion, not permission to omit that view from the decision. The `setup_board_outcome` `d_model → 512 → 9` option branch therefore declares `computation_role: independent_head`, `fusion_role: fused_input`, and `action_influence: bounded_decision_fusion`. Its causal option representation already cross-attends the current board/state, and its nine typed outputs enter a zero-safe, versioned fusion route. | Active prospectively beginning with Archaludon ex; do not hot-patch Slop Box or historical specialists. The guide still never directly selects an action or supplies a serving-time input, and its preferred action index still cannot enter the setup loss. Archaludon remains fail-closed until a checksum-bound validation receipt proves step-zero parent parity, target/mask parity, guide-target permutation invariance, nonzero branch/shared-representation and fusion-route gradients after the zero-safe warm step, bounded finite influence on fused logits, per-head leave-one-out action-logit attribution, causal option dependence, coverage/calibration, and exact role-map binding. |
| 56 | 2026-07-30 | Apply revision 55's option-conditioned action-influence rule to every learned decision head, not only `setup_board_outcome`. Each head may retain its own typed state- or option-level prediction target, but its contribution to action scoring must pass through a distinct bounded route that combines that head's output with the current legal option representation after the option has cross-attended the causal board/state. Do not average all state heads into one option-blind context that prevents distinct per-head action attribution. The current-deck guide is the sole exception: it is training-only curriculum metadata, not a learned runtime head, and has no runtime-input or action-logit route. | Active prospectively beginning with Archaludon ex through a separately versioned, zero-safe decision-fusion schema. Active Slop Box and historical checkpoints retain their current fusion bytes and behavior. Before Archaludon bootstrap or runtime activation, require exact parent parity at initialization, one declared route per learned head, nonzero finite post-training leave-one-head-out logit effects, causal suffix-invariance, legal-option dependence, bounded aggregate residuals, and a checksum-bound role/route inventory. |
| 57 | 2026-07-30 | Remove the unfinished plain `dragapult` target from the specialist-creation plan because Hammer-Pult and the completed Dragapult/Dusknoir specialist already provide sufficient Dragapult-family coverage. Replace that exact roster/order slot with `slowking`; the remaining post-Teal order is therefore `archaludon-ex`, `slowking`, then `crustle`, while the strict Hammer-Pult → Teal Mask Ogerpon ex → Archaludon ex prefix remains unchanged. Promote the already researched Slowking package into a receipt-gated specialist pre-stage using the owner's pictured Slowking/Slowpoke, Annihilape, Conkeldurr, Kyurem, Latias ex, Mega Kangaskhan ex, Smoochum, Fezandipiti ex, Meowth ex, Academy at Night, Ciphermaniac's Codebreaking, Lillie's Determination, Poké Pad, Ultra Ball, Night Stretcher, Wondrous Patch, Counter Gain, Secret Box, Psychic, Telepath Psychic, and Boomerang Energy card engine; materially different Slowking/Metagross or generic archetype lists may not substitute. Preserve plain-Dragapult corpora, routes, and audits as historical/non-planning evidence. In parallel, begin turning `docs/FINAL_MODEL_CAPACITY_AND_DECISION_REPLAY_PLAN.md` into the receipt-gated final-submission implementation plan, without activating final-model computation or disturbing the active specialist. | Active immediately in planning and preparation. It supersedes revision 34 only where Slowking was research-only and grants Slowking future roster, pre-stage, guide, training, gate, freeze, submission, dashboard-queue, and completion authority in plain Dragapult's former slot. It does not alter active Slop Box, the already staged Archaludon successor, completed Dragapult-family checkpoints, stable matchup slots, or the required-fleet total of 16. |
| 58 | 2026-07-30 | Treat Slowking as a combo/toolbox deck whose pre-stage must prove decision-head coverage for its actual multi-step lines, not only archetype classification or guide-label coverage. The checksum-bound coverage map must include top-deck construction and consumption, copied non-Rule-Box attack selection and legality, visible combo-piece/search/recovery availability, Psychic/Telepath/Boomerang attachment-return state, Wondrous Patch and Counter Gain acceleration, Slowpoke-to-Slowking and next-attacker bench continuity, opponent disruption/response, prize mapping, and remaining-turn/outcome timing. Map each requirement to independently learned causal heads with revision-56 option-conditioned action routes; require labeled-row coverage, calibration, gradients, and leave-one-head-out logit/action attribution on Slowking states. If the existing learned heads cannot materially represent one of those decision classes, add a typed combo-state head through the same zero-safe bounded fusion schema before declaring Slowking train-ready. The training-only guide may weight observed-target learning but remains the sole no-route exception and may not become an imitation policy. | Active immediately for Slowking preparation only. This does not delay or mutate active Slop Box or the already-next Archaludon specialist. Slowking registration, bootstrap, and launch fail closed until a checksum-bound combo-head coverage and validation receipt exists. |
| 59 | 2026-07-30 | Slowking's combo-specialist training may exceed the ordinary two-million-parameter target when the revision-58 coverage/ablation evidence shows that extra typed combo capacity is necessary. Treat `2,000,000` as a soft target for Slowking, not a launch ceiling. The exception is specialist-scoped, must name and count every added module, and may not weaken global or historical checkpoint validation. Retain the current `3,500,000` fail-closed ceiling for the initial Slowking design; exceeding that ceiling requires a separate owner decision and exact memory, latency, package-size, gradient-use, ablation-gain, and final-submission compatibility receipts. Unused or non-causal extra capacity is rejected. | Active prospectively for Slowking bootstrap and training only. It does not change active Slop Box, Archaludon, historical specialists, the later H10 capacity experiment, or any global environment default. |
| 60 | 2026-07-30 | Use all available training hardware for maximum sustainable wall-clock throughput. Hardware use is judged by completed exact games per second, accelerator utilization, and absence of stalls—not by maximizing the displayed simulator-slot count. The former 128-local-worker Teal topology remains rejected because it exceeded the 116 GiB cgroup and produced only a small transient public-mix rate increase before OOM/recovery loss; do not restore it blindly. Keep the active 48-worker/four-environment topology while its two local GPUs remain saturated, and permit a higher local profile only as a clean-boundary comparative trial that preserves the 100 GiB soft guard, 116 GiB hard cap, zero cgroup swap, and 12 GiB host-available floor and proves a positive exact-game wall-clock gain. Repair remote-fleet stalls as the first throughput priority: Elmo must use a managed, drain-before-exit service rotation before its observed 30 GiB process-tree RSS failure point, preserving checkpoint identity and retry-safe exact collection. | Active immediately as the throughput policy; it does not stop or restart the healthy Teal learner mid-collection. Stage Elmo's bounded rotation for the next committed no-active-remote-job boundary, validate the rendered Compose contract and endpoint/checkpoint health, and record a receipt. Retain 48 local workers unless a later boundary A/B receipt proves a faster sustainable profile; 128 is not an authorized fallback. |
| 61 | 2026-07-30 | Remove `crustle` from the required specialist-creation plan. Do not select, bootstrap, train, gate, freeze, submit, or count Crustle toward completion. Preserve every existing Crustle guide, corpus, classifier, route, stable matchup-slot, and audit artifact as historical/non-planning evidence without deletion, reindexing, or slot reuse. The complete remaining post-Teal specialist order is exactly `archaludon-ex`, then `slowking`; after Slowking completes, proceed into the already-authorized post-fleet and final-submission preparation sequence. The required specialist-fleet total is now 15. | Active immediately in planning and staged for the next receipt-backed selector/dashboard projection. Archaludon remains the hard next readiness and training gate; Slowking remains blocked behind it. This does not interrupt active Slop Box, alter completed specialists, or by itself waive the separately recorded post-fleet refresh, capacity-research, evaluation, packaging, or final-release gates. |
| 62 | 2026-07-30 | Clarify revision 61: removing Crustle as a trainable specialist does not retire its matchup-router identity. Keep Crustle's existing stable Matchup Router Format 6 slot, classifier/crosswalk identity, materialized adapter route, and causal runtime eligibility for decisions whose observed opponent state resolves to Crustle. Do not delete, disable, reindex, or reuse that route merely because Crustle is excluded from specialist selection and completion. | Active immediately as a router-preservation invariant. No checkpoint shape, active Teal behavior, selector, or service restart changes; the selector/dashboard projection must distinguish “not a required specialist” from “retained matchup route.” |
| 63 | 2026-07-30 | Use the already registered public Crustle agent as the program's Crustle practice/gate opponent instead of training a new Crustle specialist. The canonical existing binding is opponent `pilkwang-meta-20260708`, archetype `crustle` / “Crustle / Great Tusk,” source `pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-08-july`, content digest `sha256:7120bc67415e06c1cf69d64574f1a41545fd4c2fd084a029d77c5e43a357957f`. Keep it inference-only and retain the distinct Crustle matchup-router slot from revision 62; do not bootstrap, update, freeze, submit, or count it as a required specialist. | Active immediately as an identity clarification of revisions 61–62. Existing public-opponent and router evidence is reused checksum-exactly; no new Crustle training, service, selector row, or fleet count is authorized. |
| 64 | 2026-07-30 | Immediately after the exact Slowking combo specialist passes, freezes, and registers, generate and train a new separately versioned Alakazam refresh as the first model in the validated final-submission format described by `docs/FINAL_MODEL_CAPACITY_AND_DECISION_REPLAY_PLAN.md`. This Alakazam derivative satisfies the already-required post-fleet Alakazam refresh; it is not an additional specialist row and it never overwrites the immutable historical Alakazam checkpoint. Prefer a checksum-bound hot start from that existing Alakazam checkpoint when an exact compatibility migration can prove serialized-key/shape coverage, causal feature and option compatibility, zero-safe initialization of genuinely new heads/routes, step-zero behavior/parity bounds, and complete final-format role/route binding. If that migration cannot pass, preserve the failure receipt and fall back explicitly to the newest checksum-accepted core through the ordinary same-archetype Alakazam refresh path before final-format expansion; never perform a partial or cross-lineage tensor overlay. Train this Alakazam on an exact even first/second seat split, and make its eventual package choose first whenever the legal turn-order choice is exposed. Do not create a second-focused `1:7`, always-second, or second-preferring Alakazam arm under this refresh. | Active immediately for static implementation and receipt preparation. Final-format model instantiation, checkpoint migration, replay computation, training, or evaluation remains blocked until Slowking's immutable completion boundary and the required resource-isolation and release receipts. The existing Alakazam checkpoint remains unchanged, Marnie's Grimmsnarl ex remains the next refresh after the new Alakazam passes, and population training remains blocked until both refreshes pass, freeze, and register. |
| 65 | 2026-07-30 | Supersede revisions 47 and 60 only for Blackwell's local simulator-worker count: hard-stick Blackwell at exactly `128` local simulator workers and exactly `128` local games in flight. The local worker floor, target, ceiling, and live-pool maximum are all `128`; no adaptive scheduler, memory-pressure rebalance, throughput comparison, or automatic fallback may reduce that count. Preserve four environments per worker and the existing remote fleet independently. | Staged while Teal iteration 15 is active. Do not interrupt or rewrite its in-progress collection. Activate the exact 128/128 profile at the next immutable committed-iteration boundary through a checksum-bound managed receipt; after activation it remains the required Blackwell profile for Teal and every successor until a later explicit owner decision. The existing cgroup/no-swap limits remain fail-closed infrastructure guards, not authority to silently lower the worker count. |
| 66 | 2026-07-30 | Clarify the minimum lifetime of revision 65: the exact 128-worker Blackwell profile remains hard-pinned through the remaining specialist sequence, Slowking completion, and entry into the first final-submission-format Alakazam model. Reaching that model is only the earliest point at which a later explicit owner decision may reconsider the profile; it does not automatically expire, benchmark down, rebalance, or fall back there. | Staged with revision 65 for the same next immutable Teal boundary. Every specialist and refresh launch through final-format Alakazam must inherit and validate the 128/128 worker invariant. No lower profile is authorized before or automatically at that transition. |
| 67 | 2026-07-30 | Stop the in-progress Teal Mask Ogerpon ex iteration 15 now, preserve and use the latest immutable completed checkpoint at iteration 14, submit its already-required two Kaggle copies, and move immediately to Archaludon ex. Treat iteration 14 as an explicit owner ceiling acceptance without rewriting its failed measured gate into a pass. The uncommitted iteration-15 attempt may not become training or gate evidence. | Authorized for immediate managed activation. Use the user service manager and a checksum-bound stop/freeze/package/queue/handoff receipt; do not signal processes directly. Queue one `first_if_allowed` and one `second_if_allowed` bundle from the same frozen checkpoint and exact deck, keep Kaggle non-blocking, and select Archaludon only after its existing fail-closed corpus/bootstrap/route receipts pass. Revision 65's 128-worker profile applies to the next training launch. |

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
- Blackwell is hard-stuck at exactly 128 local simulator workers and 128 local
  games in flight beginning at the next immutable Teal boundary. Its worker
  floor, target, ceiling, and live-pool maximum are all 128. No adaptive
  scheduler, memory-pressure rebalance, comparison trial, or automatic
  fallback may lower the worker count. This remains mandatory through entry
  into the first final-submission-format Alakazam model and has no automatic
  expiry there. The existing cgroup, no-swap, and host memory guards remain
  fail-closed infrastructure protections and may stop a failed attempt, but
  they may not silently substitute a smaller profile.
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
- Every training-complete specialist creates the checksum-bound, single-use
  Kaggle submission authorizations required by its canonical submission
  profile automatically. The default is exactly one first-preferring copy;
  Teal Mask Ogerpon ex requires exactly two copies, one first-preferring and
  one second-preferring, from the identical frozen checkpoint and deck.
  Submission remains asynchronous and non-blocking, and delayed copies always
  retain the original frozen passing checkpoint identity.
- The four-hour Kaggle spacing window is anchored to the second-most-recent
  logical submission after deduplicating its reconciled Kaggle and queue rows;
  the newest submission alone never delays the next queued copy.
- Expanded-head retrofits use new derivative identities and never rewrite,
  replace, re-freeze, or silently promote an original passing checkpoint.
- Retrofitted heads remain dormant and runtime-disabled until their own
  training, validation, checksum registration, and activation receipt exist.
- The active specialist's expanded strategic heads receive gradients in every
  full-model RL or rehearsal update for which their exact causal labels are
  present; missing labels remain masked.
- Each newly prepared specialist has exactly one checksum-bound current-deck
  guide contract and generic training-only guide curriculum. Guide evidence is
  cited, signals are causal and specialist-specific, missing signals are
  masked, and no guide has a runtime-input or action-logit route.
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
- Every specialist training run started under the prospective revision-44
  policy uses the realized-win importance lifecycle: rapid bootstrap and
  evidence-backed post-bootstrap ramp, a positive-contribution plateau, then
  evidence-driven decay toward zero. Generic controllers may adjust those
  future runs' auxiliary weights only within `0.00`–`0.50` and only at a clean
  checksum-receipted boundary. Completed, frozen, and already-started runs are
  never backfilled or rewritten; active Teal retains only its explicit
  revision-42 `0.25` correction.
- For already-started legacy guide runs, including active Slop Box, the
  receipt-backed value remains the literal multiplier on masked,
  confidence-weighted guide policy cross-entropy. For future runs beginning
  with Archaludon ex, revisions 51 and 56 supersede that target: the same
  bounded multiplier scales only guide-conditioned losses on observed causal
  learned-head targets and direct guide-to-policy cross-entropy is
  forbidden. Ramp ascent is evidence-governed learning pressure, not an
  elapsed-time-only counter or a guide-head/dashboard statistic. A review
  records the earliest eligible next iteration; its nonblocking evidence is
  applied at the first later available five-iteration hard pause. The boundary
  receipt records the actual iteration. No guide weight is a serving-time
  action bias.
- Every future specialist beginning with Archaludon ex carries a typed
  head-role map. Each strategic head is computed independently from the causal
  board state and, where option-conditioned, the current legal option, and
  every learned decision head has a distinct bounded decision-fusion route
  that combines its typed output with the current board/state-cross-attended
  legal-option representation. State-level prediction targets remain valid,
  but their action contributions may not be collapsed into one option-blind
  averaged context. The current-deck guide is the sole exception because it is
  training-only curriculum metadata, not a runtime head. The
  `setup_board_outcome` branch is `computation_role: independent_head`,
  `fusion_role: fused_input`, and
  `action_influence: bounded_option_conditioned_route`. It trains only on selected
  complete setup-stage rows with observed next-state/outcome targets. Its
  guide-conditioned term depends on confidence and label presence, never the
  guide's preferred action index. A zero-safe migration may begin with exact
  parent behavior, but runtime activation requires measured nonzero,
  finite, per-head action-logit influence and leave-one-head-out attribution.
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
- Hammer-Pult is declared nonlinear. Its pre-stage receipt is invalid for
  bootstrap unless it checksum-binds all required nonlinear decision systems,
  the authoritative 17-input fused policy, exact causal masking, and
  training-ineligible guide-on/guide-off annealing evidence.
- A completed auditable Hammer-Pult iteration-15 performance-gate failure uses
  the latest exact checkpoint, preserves the failed result, queues the normal
  Kaggle copy, and proceeds to Teal as `ceiling_accepted`; invalid or missing
  checkpoint/receipt evidence still fails closed.
- After Spidops, the strict successor prefix is Hammer-Pult, Teal Mask Ogerpon
  ex, then Archaludon ex. A missing strict-prefix input blocks fallback and
  triggers recovery.
- Plain `dragapult`, `dragapult-blaziken`, `dragapult-dudunsparce`, `crustle`,
  and `walrein` are not required
  specialist targets and never count toward program completion. Historical
  artifacts and stable matchup-slot evidence remain immutable and non-planning.
- Crustle remains an active causal matchup-router identity despite being
  excluded from specialist training. Its Matchup Router Format 6 slot,
  classifier/crosswalk identity, and materialized adapter route remain
  eligible when observed opponent state resolves to Crustle; the route may
  not be disabled, reindexed, deleted, or reused as a consequence of revision
  61.
- The inference-only public opponent `pilkwang-meta-20260708` is the canonical
  Crustle practice/gate agent. Its exact registered package digest is reused;
  it receives no gradients and is not a specialist checkpoint, roster row,
  completion credit, or submission candidate.
- The remaining post-Teal specialist order is exactly Archaludon ex, then
  Slowking. Slowking occupies plain Dragapult's former required-fleet slot, and
  its exact pictured card engine—not a generic Slowking variant—owns its guide,
  representative, corpus filter, and pre-stage identity. After Slowking,
  proceed to the separately gated post-fleet/final-submission preparation
  sequence; there is no third specialist fallback.
- Slowking is a combo/toolbox specialist. It cannot become train-ready until
  its receipt proves causal learned-head and bounded action-route coverage for
  top-deck construction, copied-attack choice, combo pieces, energy movement,
  recovery, bench continuity, disruption response, prizes, and outcome timing.
- Slowking may exceed the ordinary two-million-parameter soft target for
  evidence-backed combo capacity, but its initial specialist-scoped hard
  ceiling remains 3.5 million with exact module accounting and resource,
  causality, gradient-use, ablation, and package receipts.
- The required fleet remains exactly 15 specialists. The post-fleet Alakazam
  and Marnie's Grimmsnarl ex refreshes are separately versioned derivatives,
  not new roster rows and not replacements for their immutable historical
  passing checkpoints.
- Once all 15 required specialists are frozen, run the refreshes strictly as
  Alakazam then Marnie's Grimmsnarl ex. The Alakazam refresh is the first
  final-submission-format model and starts immediately after Slowking's
  checksum-bound completion. Prefer a validated hot start from the immutable
  existing Alakazam checkpoint; if that compatibility migration fails, record
  the failure and use the newest checksum-accepted core through the ordinary
  same-archetype refresh path before expanding to the final format. Resolve all
  current canonical training structures at each refresh start; never pin
  either refresh to today's core version, compatibility derivative, schema
  digest, or runtime package, and never partially overlay tensors from
  incompatible lineages.
- The post-Slowking final-format Alakazam refresh uses an exact 50/50
  first/second training-seat split. Its production package preference is
  `first_if_allowed`; the optional second-focused `1:7` curriculum,
  always-second behavior, and second-preferring package do not apply to this
  Alakazam refresh.
- Its preferred immutable hot-start checkpoint is the accepted Alakazam
  checkpoint
  `sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea`.
  Direct migration must either validate completely or fail without producing a
  promotable partial child.
- Population training remains blocked until both post-fleet refresh versions
  independently pass, freeze, and register. Before that boundary neither
  refresh has current selector, pre-stage, runtime, service, or gradient
  authority.
- V6 permits at most one matchup slot per decision; unknown, retired,
  ambiguous, or insufficiently observed matchups exactly bypass the bank.
- Router Format 6 slot identities never move and retired slots are never automatically
  reused.

## Safe resume

Read this file first, then read the referenced machine protocol, mutable state,
slot registry, runtime selector, and latest immutable receipts. Continue the
healthy managed workload if their evidence agrees. If a staged change is
pending, finish its validation in parallel and apply it only at its declared
safe boundary.
