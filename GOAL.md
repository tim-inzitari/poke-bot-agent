# Pokémon RL Goal Gateway

Schema: `poke_bot.goal_gateway/v1`  
Revision: `310`

Status: `authoritative`

This file is the stable entry point for every resumed controller. The product
goal should remain short:

> Continuously execute the authoritative contract in `GOAL.md`. Record every
> explicit owner design change immediately, update and validate the referenced
> canonical sources, preserve all non-conflicting verified progress, and
> activate changes at the next safe receipt-backed boundary unless the owner
> explicitly orders immediate activation.

## Current objective

### Active task routing

The active Alakazam rule-derivative task is delegated to
`goals/alakazam-elmo-rule-derivative/GOAL.md`. Its sole typed authority is
`goals/alakazam-elmo-rule-derivative/contract.json`, and its current
receipt-backed phase/artifact projection is
`goals/alakazam-elmo-rule-derivative/STATUS.json`. The status file is
presentation and resume state only: it may not authorize work, override the
dedicated contract, or replace immutable receipts.

This root gateway and `state/alakazam-new-list-direct-policy-r241.json`
continue to own the separate r274 terminal lineage and the production handoff
boundary. Reading `/goal` for the derivative must follow the delegated files
above; it must not merge derivative semantics or status into r274 ownership.

Treat Slowking as a terminal failed experiment under the explicit revision-79
owner boundary. Preserve its sealed collections, failed evaluations, and
uncommitted iteration-6 work as immutable audit evidence, but do not freeze,
register, submit, or serve any Slowking checkpoint. Proceed immediately to a
separately versioned, up-to-date Alakazam refresh in the validated
final-submission format. Prefer a checksum-bound
compatibility hot start from the immutable existing Alakazam checkpoint when it
can pass the required migration and parity receipts; never rewrite that parent.
Continue the final-format Alakazam refresh through the durable zero-indexed
`iter_00020` boundary. Do not collect `iter_00021`. At that boundary, freeze
and register the exact iteration-20 checkpoint: record a normal measured pass
when every existing gate passes, otherwise record an explicit owner ceiling
acceptance while preserving the failed measured-gate evidence and never
calling it a pass. Then begin the separately versioned, up-to-date Marnie's
Grimmsnarl ex refresh and continue it through the durable zero-indexed
`iter_00020` boundary without collecting `iter_00021`. After that exact
checkpoint is truthfully completed, frozen, and registered, a new separately
versioned H10 Crustle specialist was started; under revision 170 the owner
abandons Crustle for now (preserve all artifacts; no deletes) and immediately
activates the separately versioned H10 + RTP Slop Box specialist
(`teal-mask-ogerpon-ex` / Raging Bolt Ogerpon), distinct from historical
Teal/Slop Box, under the dual-pipeline, Cox/Chao guide, full-archetype expert
bootstrap, ≥90% Cox/Chao policy-accuracy gate, deck-agnostic Alakazam/Marnie
warm-start, and expert-bootstrap RTP-cut contracts. Under revision 171, when
Chao-hard CE overfits again—train ≫ held with Cox/Chao held stuck ≪0.90
(do not keep spinning for a measured 0.90)—the owner authorizes explicit
ceiling proceed: select best Chao-held checkpoint, record measured fail
evidence, mark ready via owner ceiling (never call 0.90 a pass), queue a
nonblocking `first_if_allowed` Kaggle milestone with the Cox/Chao deck (RTP
cut sidecar if ready), then register and start the Slop Box H10
self-play/public-mix RL loop with expert rehearsals every 5 iterations.
Ceiling acceptance must preserve measured gate evidence and must never be
labeled a measured pass. Under revision 172, Slop Box's strong-public /
formal premium-holdout roster must include the abandoned non-active H10
Crustle specialist as tier `S` weight `2.0` (r111-style), bound to the
committed iter_00004 milestone package—not incomplete iter5 quarantine.
Under revision 173, Slop Box's retained H10 `combo_state` head receives a
separately versioned target schema `poke_bot.slop_box_combo_state_targets/v1`
that maps Teal Dance / Crispin / Glass Trumpet / Energy Switch / engine
continuity into the existing 32-d head without width remap; expert-trajectory
and live-RL attach paths must emit nonzero masks, and ordinary combo loss
weight remains `0.025`. Jul31–Aug5 remat repair continues in parallel
and folds into later rehearsals if not ready before RL start. Under revision
174, the fresh Slop Box H10+RTP expert bootstrap is owner-capped at outer
epoch 40 (not 300): leave the healthy live CE alone until the epoch-40
checkpoint boundary, stop cleanly via systemd (no mid-epoch kill), then
queue/submit one nonblocking `first_if_allowed` Kaggle milestone with the
Cox/Chao deck lineage from submission 55188658, then lift the RL hold only
after H10+guide preflight and start self-play/public-mix RL (Crustle S-tier
roster retained; wire matchup adapters from the marked corpus). Under revision
175 the owner hard-swaps active training to a separately versioned Alakazam
RTP loop: pilot the Abra/Kadabra/Alakazam/Dunsparce archetype list, expert
refresh + CE rebootstrap from the Alakazam checkpoint on the last five days
of Alakazam data (2026-08-01..05), Kaggle submit, then repeating self-play
(1024 mirrors, fill to 8196 public-mix games, ≥1024 Grimmsnarl/set pinned to
`sha256:f20efb20f5c3…` / `specialist-marnie-final-format-h10-f20efb20f5c3`),
with expert refresh + Kaggle every 5 iterations, iteration ceiling 300, all
non-combo heads live/nonzero (guide ON; combo head OFF). Jul24–Aug5 Slop Box
CE recovery stays held/off. Preserve Crustle's
stable Matchup Router Format 6 slot and `pilkwang-meta-20260708` as
inference-only baseline/history. Transition to population round-robin only after
Alakazam, Marnie's Grimmsnarl ex, and Slop Box H10 RTP are truthfully
training-complete, frozen, and registered (Crustle may rejoin later by explicit
owner order; it is not a current population blocker while abandoned).

Under revision 176, the active standalone read-only localhost replay/model
inspector is separate from the training dashboard and its service. It indexes
Elmo's checksum-backed downloaded Kaggle submission replay cache, resolves each
submission/game/decision to the exact immutable bundle and checkpoint, and can
re-run the recorded acting model at any causally reconstructable step. The UI
must expose the observation and legal options, raw and normalized values for
every architecture-present head, masks, Fusion-v3 route reliabilities and
per-option contributions, final logits/probabilities and selected/recorded
actions, plus checkpoint parameter names, shapes, statistics, norms, and
bounded tensor slices. Every displayed value carries provenance and an
explicit availability reason when older replay or model formats cannot supply
it. The inspector is localhost-only, lazy-loads one checksum-verified model at
a time under bounded resources, never mutates replay/checkpoint bytes, never
changes the selector or managed services, and never makes Kaggle/evaluation
replays training-eligible. Typed contract:
`state/replay-model-inspector-owner-design-r176.json`.

Under revision 177, the inspector remains a separately managed read-only
service bound only to Elmo loopback, while the dashboard gains one
presentation-only link that opens it through the dashboard's existing HTTPS
external-access edge. The gateway is restricted to the fixed
`/replay-inspector/` prefix, reuses the dashboard edge's existing access
policy, permits read-only GET requests only, and reaches Elmo through a
separately managed encrypted Bert-loopback-to-Elmo-loopback tunnel. It must not
bind the inspector to a LAN or public interface, forward browser credentials to
the inspector, add dashboard API or training authority, or make evaluation
replays training-eligible. Typed contract:
`state/replay-model-inspector-dashboard-gateway-r177.json`.

Under revision 178, the same dashboard link must also work from the direct LAN
dashboard when the router cannot hairpin the public hostname. The link is a
same-origin relative `/replay-inspector/` path. Bert's dashboard listener may
proxy only that fixed prefix to the existing Bert-loopback tunnel, only for
loopback/private clients and read-only GET requests, with strict path
validation, upstream pinning, cross-site rejection, and browser credential and
origin stripping. This transport-only LAN gateway does not merge inspector
state or authority into the dashboard; Elmo's inspector and Bert's encrypted
tunnel remain loopback-only, the authenticated external Caddy route remains
unchanged, and evaluation replays remain training-ineligible. Typed contract:
`state/replay-model-inspector-lan-gateway-fix-r178.json`.

Under revision 179, the inspector's primary decision view is human-readable
rather than tensor-first. Every legal option must include a truthful plain-
English transcript of the action at its factorized stage while retaining its
raw representation for audit. Every architecture-present head must also show
how removing that head changes the final policy: direction and magnitude for
the selected option, the largest option-probability shift, and the legal
actions it most helps and hurts. These values must come from the exact causal
leave-one-head-out final-policy recomputation, not from raw head magnitude or
invented historical telemetry; unavailable formats retain explicit reasons.
Typed contract:
`state/replay-model-inspector-human-readable-analysis-r179.json`.

Under revision 180, submission selectors and headers must pair every numeric
submission ID with its exact source-backed submission text or label. Game
metadata must show both players' available names, seats, and leaderboard ranks.
Ranks and labels may come only from the archived Kaggle metadata or an exact
submission-bound provenance record; absent fields render as explicitly
unavailable and are never guessed from score, reward, filename, or opponent
name. Typed contract:
`state/replay-model-inspector-submission-player-context-r180.json`.

Under revision 181, every inspected decision must display one prominent plain-
English Matchup Adapter state: active for this decision, bypassed with a causal
reason, or unavailable with a reason. An adapter is active only when the
decision-specific runtime actually selects and applies a matched route—not
merely because adapter weights exist in the checkpoint. When active, show the
source-backed matchup/archetype and slot or route identity, reliability, and
exact policy effect when available. Typed contract:
`state/replay-model-inspector-matchup-adapter-status-r181.json`.

Under revision 182, the active Alakazam RTP r175 collector uses true four-game
`LibcgMultiEnv` packing for only the 27 checksum-exact public packages proven
retention-compatible in attempt 0040. Eligibility is default-deny and binds the
exact opponent ID, package content digest, portable-baseline contract, and
training group in both the job spec and target provenance. The 10 legacy gate
packages that caused the attempt-0040 shortfall always use isolated one-game
remote `play`; unknown, changed, malformed, or cross-group packages do too.
Self-play remains packed at four. Public dispatch keeps RTP and both remotes
engaged (`PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`) and opens a public-only second
request wave so one pack can execute and one can wait per remote worker,
without doubling self-play queue depth. Every packed child retains independent
identity, result, execution credit, and exact-retention accounting. Do not
weaken the exact `1024 + 7172 = 8196` collection contract or its 32 deterministic
public replacement lanes. Typed contract:
`state/alakazam-public-multi-env-split-r182.json`.

Under revision 192, the exact H10 Marnie's Grimmsnarl ex package
`specialist-marnie-final-format-h10-f20efb20f5c3`, bound to checkpoint
`sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381`,
is an explicit additional Alakazam r175 specialist opponent at tier `S++`.
It remains a distinct row alongside the historical old-format Marnie
specialist and may never be collapsed into, substituted for, or relabelled as
that historical identity.  Scoped `S++` means weight `4.0` relative to tier-A
weight `1.0`; the existing checksum-exact minimum of 1,024 H10 Marnie games
per 8,196-game set remains mandatory, and the exact `1,024 + 7,172 = 8,196`
collection total is unchanged.  Revision 182 has already resumed a healthy
iteration-1 process, and r175's configured hard boundary pause does not begin
until terminal-eligible iteration 5. The staged candidate is not armed: the
current source has no trainer-owned handoff fence between that pause and its
unconditional next-collection dispatch, so a managed stop could race into
iteration 6. Activation waits for either a receipt-proven inactive boundary
or a later checksum-bound fence-enabled source; observing the pause alone
never authorizes a stop or restart. Continue the healthy parent without
interruption while roster, runtime registry, dispatch provenance, and focused
exact-retention tests are validated. Because revision 182 admitted this
package to packed transport only under its former `diverse_public` provenance,
its legacy diverse-public pin is removed at activation and the exact active-gate
strong-practice row becomes the sole executable 1,024-game floor.  The new
strong-practice rows default to singleton remote play until that exact
group binding receives a separate retention attestation; all other r182
eligibility remains unchanged.  Typed contract:
`state/alakazam-marnie-splusplus-opponent-r192.json`.

Under revision 193, the owner orders one large expert refresh for the active
Alakazam r175 lineage followed by one checksum-exact Kaggle resubmission. Do
not interrupt the healthy in-flight iteration 14. At the first receipt-proven
inactive or trainer-owned fenced boundary after iteration 14 is durably
committed, arm the one-time override before iteration 15 collection begins.
After iteration 15's exact collection is sealed and before its RL optimizer,
train exactly 25 expert epochs (five times the ordinary five-epoch rehearsal)
from the latest durable continuous-learner checkpoint against the checksum-
pinned 2026-08-01..05 Alakazam expert corpus. This is a full-model refresh:
every architecture-
present non-combo learner head remains trainable, the strategic-directional
current-deck guide stays training-only at weight `0.05`, setup-board outcome
stays at `0.025`, and combo loss and combo fusion route remain disabled. The
refresh must emit an immutable checkpoint and receipt binding the parent,
corpus, optimizer schedule, completed 25/25 epochs, validation metrics, model
structure, and SHA-256. Build exactly one `first_if_allowed` Kaggle bundle from
the durable iteration-15 checkpoint whose training provenance checksum-binds
that 25-epoch refresh, using the exact r175 pilot deck; fail closed unless
the submitted runtime, matchup tree, adapter-bank activation, deck, checkpoint,
smoke/parity results, and single-use submission authorization are checksum-
bound. Queue/upload it under the existing quota and spacing rules without
making Kaggle availability a reason to lose the refresh or falsify a
submission. Resume the ordinary five-iteration/five-epoch cadence afterward.
This owner refresh does not change the exact `1,024 + 7,172 = 8,196`
collection contract, does not declare a measured gate pass or final freeze,
and does not authorize a pause marker by itself to stop a live trainer. Typed
contract: `state/alakazam-large-expert-refresh-resubmit-r193.json`.

Under revision 194, submit one additional checksum-exact Kaggle copy of the
durable Alakazam iteration-15 checkpoint
`sha256:c2b01f5a12a4164e282f278e104da3dd5d5b0c1467d592e01d832be141fcf69c`.
The second copy uses the already verified revision-193 bundle bytes and r175
pilot deck, remains `first_if_allowed`, and receives its own unique label,
queue identity, one-shot authorization, and upload receipt. It does not replace
or rewrite submission `55359777`, does not alter the checkpoint or runtime,
and must not pause or restart healthy training. Submit immediately under the
existing daily quota and spacing rules. Typed contract:
`state/alakazam-iter15-second-copy-r194.json`.

Under revision 195, focus exclusively on the completed Alakazam lineage and
run one additional large expert bootstrap from the immutable terminal
iteration-20 checkpoint
`sha256:87caf05bdeda3a798268905a5670841125b1797f31b9a823343c393d7f0ced65`.
The bootstrap is exactly 25 full-model expert epochs over the checksum-pinned
2026-08-01..05 Alakazam corpus. Every architecture-present non-combo learner
head remains trainable, the strategic-directional guide remains training-only
at weight `0.05`, setup-board outcome remains `0.025`, and combo loss and the
combo fusion route remain disabled. The new checkpoint and receipt are
immutable derivatives; never rewrite the terminal checkpoint, its registration,
or either revision-193/194 submission. Build and submit exactly two new
`first_if_allowed` Kaggle bundles from the same refreshed checkpoint and exact
r175 pilot deck so the only intended serving difference is RTP. Copy 1 must
disable RTP completely in the submitted runtime: no recursive-turn-planner
activation, no RTP checkpoint environment, no packaged RTP sidecar, and no
startup path that can enable RTP. Copy 2 must package and checksum-bind the
canonical Alakazam r175 RTP sidecar and force recursive-turn-planner activation
from the submitted entrypoint. Matchup Adapter runtime remains required under
revision 185 for both copies when the trained bank is present. Bundle inspection
and submitted-entry smokes must prove copy 1 acts without RTP and copy 2 acts
with RTP. Their unique Kaggle submission label/messages must visibly include
the exact text `NO RTP` and `RTP`, respectively. Queue/upload both under
the existing quota and spacing rules, preserving a pending receipt if Kaggle
cannot accept it immediately. This completed as submissions `55378392`
(`NO RTP`) and `55378477` (`RTP`), both from checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`;
the latest source-backed public scores are `500.4` for `NO RTP` and `600.0`
for `RTP`. Slop Box CE/RL holds remain in place and receive
no gradient or runtime authority from this order. Typed contract:
`state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json`.

Under revision 196, preserve the exact submitted package for every accepted
owner Kaggle submission on Elmo's NAS, not only its downloaded replay bytes.
Backfill every submission currently indexed from the permanent historical
special case through the newest automatically discovered owner submission.
For queue-backed submissions, archive the checksum-verified uploaded bundle,
its exact packaged runtime tree, checkpoint/model bytes, matchup tree, label,
and immutable submission-ID binding under content-addressed paths. Future
accepted queue rows receive the same archival automatically. The inspector may
enable dynamic reconstruction only after the exact package/runtime parity is
attested; missing or conflicting historical bytes remain replay-only with an
explicit reason and must never inherit a nearby checkpoint. This archival is
read-only with respect to training and does not make evaluation replays
training-eligible. Typed contract:
`state/replay-model-inspector-all-submission-artifact-archive-r196.json`.

Under revision 197, rebuild Alakazam RTP as a separately versioned,
receipt-backed production candidate instead of tuning or overwriting the
revision-195 sidecar. Preserve the terminal r175 checkpoint and registry, both
revision-195 bundles and submissions, and sidecar
`sha256:dde7b813e69cabc9c3ad0c3c24eedfc85f05469cf739697c818111bb7acc3aee`
as immutable evidence. The aligned candidate binds the exact revision-195
policy parent
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
the protected 2026-08-01..05 Alakazam expert corpus, a whole-game
source-disjoint heldout split, and the same complete ordered legal-action space
used by the serving bridge. The new `pure_rl_r197` profile is distinct from and
does not rewrite legacy `pure_rl`. Its four-candidate, depth-two planner has an
owner-selected initial hard ceiling of exactly 32 neural passes; serving
preflight must also prove the current path's six-pass requirement fits that
ceiling. A separately versioned profile may raise the ceiling only when a
measured required-pass and latency receipt justifies it, and never above the
absolute owner bound of 256; escalation is never automatic or unbounded. Train
outcome/value and calibrated value-of-planning targets without fabricating
counterfactual labels for unobserved actions; a candidate without trustworthy
alternative-action targets remains shadow-only. The checkpoint, data,
configuration, code, deck, matchup tree, and promotion evidence are all
checksum-bound, and serving load fails closed on any identity, eligibility,
action-authority, schema, state-key, or pass-budget mismatch. Evaluate frozen
NO-RTP, direct-bridge-only, and recursive-RTP arms so bridge effects are not
credited to recursion. Recursive action authority requires nonzero verified
planner use, zero neural-pass-budget failures, bounded fallback/latency and
legality evidence, source-excluded heldout improvement of recursive RTP over
the direct bridge, and a separate immutable promotion receipt. Do not restart
the terminal r175 trainer, collect iteration 21, mutate the canonical selector,
or authorize a new Kaggle submission merely to build the shadow candidate.
Typed contract: `state/alakazam-rtp-realignment-r197.json`.

Under revision 198, before any revision-197 corpus or candidate is
materialized, supersede its two sizing values: enumerate up to exactly 1,024 complete ordered legal-action combinations
per decision, and set the
`pure_rl_r197` planner's hard neural-pass ceiling to exactly 256. The current
recursive skeleton still requires only six passes in its normal path and five
for forced replan; 256 is available headroom, not a required amount of work.
There is no automatic escalation above 256. Reissue every corpus, planner,
runtime, package, evaluation, configuration, and receipt identity from these
values; no artifact or receipt built for the superseded 256-action/32-pass
draft is eligible. The candidate remains shadow-only and all revision-197
non-regression and promotion gates remain unchanged. Typed contract:
`state/alakazam-rtp-realignment-r197.json`.

Operational reconciliation for revision 198: the first two production shadow
materialization attempts failed closed before publishing a corpus, candidate,
sidecar, or training receipt. Preserve both content-addressed source roots as
immutable evidence and never retry either one in place. Source root
`alakazam-rtp-r197-src-8ea613975e10` failed pre-start verification after an
ad-hoc post-publication import created unlisted Python bytecode. Source root
`alakazam-rtp-r197-src-89eef1d25f9f` passed source and runtime preflight, then
stopped during complete-action conversion because the current global rule
classifier returned `unknown` for protected Alakazam identity `89228866`, seat
1, and the visual converter consequently emitted no matching acting-seat
record. At that failed boundary the dedicated r197 output contained only an
empty physical corpus parent. A later attempt was permitted only from a new
clean content-addressed source root with a fail-closed protected-identity
conversion repair that re-verifies the exact raw episode, seat, and deck,
permits only an identity-local `unknown`-to-`alakazam` label for that exact
protected seat, and rejects every recognized conflicting archetype. The exact
1,024-action/256-pass shadow-only contract and every r175/r195 preservation,
selector, promotion, and Kaggle restriction remain unchanged.

Revision-198 production shadow completion: source attempt 3 used the new,
verified content-addressed root `alakazam-rtp-r197-src-2ae56bc6a2db` and the
managed `pokebot-alakazam-rtp-r197-shadow.service` completed successfully with
zero restarts at `2026-08-09T21:30:09Z`. It sealed complete-action corpus
`r197-complete-actions-b534410c2511272dd2af38d71b40196d9566e41936e41d3f0797334d2713c157`
with 377,493 training-eligible complete-action rows and 3,643 explicit
over-cap nontraining audit rows, then emitted shadow candidate
`r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e`
and sidecar
`sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31`.
Independent strict verification re-opened every bound artifact, safely loaded
the sidecar, and reproved the exact 1,024-action / 256-pass contract with six
normal and five forced-replan neural passes. Training and heldout losses were
`1.439791690807529` and `1.4853265887337121`; trustworthy alternative-action
return, ranking, and calibration targets remained absent and were masked
rather than fabricated. The candidate therefore remains shadow-only:
serving, action, selector, checkpoint-publication, submission, and promotion
authority are all false. A true-randomness-matched NO-RTP/direct-bridge/
recursive-RTP evaluation, runtime reliability and latency gates, heldout
recursive-over-direct improvement, and a separate accepted promotion receipt
remain required. No r175 restart, iteration 21 collection, selector change, or
Kaggle submission occurred.

Under revision 199, the owner rescinds the provisional instruction to abandon
RTP and continues Alakazam RTP as separately versioned, receipt-backed,
shadow-only research and development. The poor preliminary prefix from r198
attempt 10 is non-terminal telemetry, not an abandonment, efficacy, promotion,
or stop decision. Preserve the already-running attempt unchanged and allow it
to reach an immutable terminal evaluation boundary; do not preempt, restart,
alter, or retry it in place. Preserve its complete evaluation, review, HOLD,
rejection, or fail-closed receipts regardless of outcome. A HOLD, rejection,
invalid attempt, or failed gate retains zero RTP serving, action, selector,
checkpoint-publication, submission, and promotion authority and permits only a
new content-addressed, separately versioned shadow candidate after a
receipt-bound diagnosis and preflight. Do not invent an abandonment threshold
from a live prefix or weaken the complete-action, 1,024-action, 256-pass,
six-normal/five-forced-replan, paired-evaluation, reliability, latency,
legality, source-excluded recursive-over-direct, trustworthy-counterfactual,
or separate-promotion gates. Preserve all earlier source and evaluation
attempts plus r175/r195; do not restart r175, collect iteration 21, change the
selector, or submit to Kaggle. Typed continuation contract:
`state/alakazam-rtp-continuation-r199.json`.

Under revision 200, the owner directs continued RTP improvement and permits a
different GPU turn-planning strategy. Develop a new, separately versioned,
conservative batched one-turn complete-action GPU reranker while r198 attempt
10 continues unchanged. The new planner must score only the current legal
complete-action set, must never execute a stale multi-action program, and must
default exactly to the frozen base-policy action unless separately receipted
counterfactual training, calibration, uncertainty, support, and value-margin
gates all permit an override. The r197 candidate's zero trusted
counterfactual-return, ranking, and calibration targets cannot be relabelled as
such evidence. Any new targets must come from checksum-bound, paired simulator
branches over source-excluded games, use policy-visible information only, and
remain masked when unavailable; evaluation, Kaggle, and hidden-opponent data
remain training-ineligible. GPU batching and microbatching must be numerically
equivalent, bounded, state/cache/RNG-pure, and legality preserving. This is
offline and shadow research only: no current service, selector, serving,
action, checkpoint-publication, submission, or promotion authority changes.
Any executable candidate requires a new content-addressed source, sidecar,
candidate, evaluation identity, output root, independent latency/reliability
evidence, and separate accepted promotion receipt. Typed research contract:
`state/alakazam-gpu-turn-planner-r200.json`.

Under revision 201, the owner clarifies that the GPU planner must plan one
whole current turn with multiple atomic steps, not merely rerank one action.
This supersedes revision 200's single-step design before any module, candidate,
service, or authority was activated. Develop a separately versioned,
closed-loop receding-horizon full-turn planner: at every decision it may score
bounded candidate trajectories through the current turn's typed `END_TURN`,
but it executes only the next atomic action, consumes the resulting real
policy-visible observation and legal-action set, and replans the remaining
turn. It must never plan across turns, blindly default a missing branch
predicate, reuse stale root legal actions or option encodings, or execute a
cached remainder whose expected observation/legal-action fingerprint changed.
The frozen direct-policy action is a mandatory candidate and remains the exact
fallback unless trusted multi-step transition, counterfactual return/ranking,
calibration, uncertainty, support, legality, latency, reliability, and
positive-margin evidence authorizes each next action. A valid successor-state
and future-legality mechanism is a prerequisite; the battle-start pairing
snapshot is not an arbitrary mid-game branch API, and the current latent
rollout cannot be relabelled as an exact beam or simulator. Preserve attempt 10
and every predecessor unchanged. This is offline/shadow research only and
grants no training-service, evaluation-service, serving, action, selector,
checkpoint-publication, submission, promotion, r175-restart, or iteration-21
authority. Typed research contract:
`state/alakazam-closed-loop-turn-planner-r201.json`.

Under revision 202, the owner supersedes revision 201's unconditional
replanning and current-turn-only limits before a phase-1 module or any runtime
authority was activated. Develop a separately versioned chance-aware cached
inter-turn expectimax/MCTS planner. It still executes only one atomic action
before checking the real policy-visible observation, complete legal-action
set, option encoding, turn key, and immutable subtree identity, but it reuses
the exact matching deterministic subtree without recalculating it. A turn-key
change is not by itself invalidating when an information-set-safe successor
and exact future legality were precomputed and all fingerprints match.
Unknown, hidden, unbounded, or incompletely modelled randomness and opponent
information are hard search boundaries. A simple finite chance point such as
a fully enumerated fair coin flip may be expanded only when every outcome and
exact rational probability is known, every outcome has a valid successor and
future legal set, and the backed-up value is exactly the probability-weighted
sum of child values; never sample a subset, determinize hidden state, or
reweight the probabilities and call it exact. This permits the search to value
states beyond a simple chance point, but every realized chance outcome remains
an execution-cache boundary and begins a fresh public root; only a no-chance
deterministic transition may advance the cached execution subtree without
recalculation. The planning budget lives in one easily changed typed
configuration object with defaults of 20 seconds total planner wall time per
actual turn and 5 seconds before any one atomic action. Both are hard
fail-closed ceilings, all planner work is charged, and a timeout returns the
exact direct-policy action without granting a partial tree authority. Any
changed effective values require a new configuration identity before runtime
evaluation. The current r197 data still has no trusted multi-step,
counterfactual-ranking, or calibration authority; MCTS/expectimax quality and
runtime action authority remain forbidden until the exact successor,
future-legality, chance-distribution, target, calibration, support, latency,
and reliability receipts exist. Attempt 10 and all predecessor evidence remain
untouched. Typed research contract:
`state/alakazam-chance-aware-inter-turn-mcts-r202.json`.

Under revision 203, every currently indexed accepted owner submission has a
checksum-attested exact submitted runtime and a trace-ready replay index. A
recorded setup prompt that the submitted entrypoint answers before neural
initialization remains inspectable: show the exact deterministic runtime
distribution (`100%` for the selected legal answer and `0%` for every other
answer), mark it as an exact runtime short circuit, and explicitly state that
these are not neural-softmax probabilities. Such a prompt has no neural
logits, value, strategic-head influence, fusion route, or matchup-adapter
effect; never fabricate those fields. A separately labeled hypothetical
neural rerun may score that same setup observation with the exact archived
model, but must never be described as historical Kaggle execution. Ordinary
model decisions run checksum-bound forward passes on Elmo's GPU. Submission
and game lists are searchable, exact step numbers are directly addressable,
and selecting a game warms every recorded step/stage sequentially into a
device-local cache after rendering the initially selected step. Typed
activation source:
`state/replay-model-inspector-deterministic-runtime-policy-r203.json`.

Under revision 204, the Replay Model Inspector adds a nontechnical Head FAQ
covering every one of the 19 Fusion policy inputs. Each entry states the
ordinary-language question the head was trained to answer, its time horizon,
and its important interpretation caveat. In particular, `lethal_threat` is
identified as our near-term offensive prize-taking signal rather than our
death risk; game-level loss uses the outcome-distribution/value stack, and
near-term knockout danger uses opponent-response and tactical-outcome signals.
For the selected decision, each FAQ entry also displays the exact causal
leave-one-head-out policy effect and baseline `1x` setting, never a fabricated
fixed percent importance; nonlinear head effects are not additive. The same
trace may compute a checksum-bound current-deck guide recommendation as a
production shadow diagnostic. That guide remains a training teacher rather
than a learned policy head: it has exactly zero logit delta, cannot select or
override the submitted action, and is unavailable rather than guessed when no
single exact-runtime guide safely recognizes the deck. A visible off/on
comparison toggle reveals or hides the guide recommendation and explicitly
reports whether its preferred action matches the model. It names both actions
and states what action the guide would hypothetically choose if it controlled
the decision; both toggle positions retain the same submitted-model logits and
probabilities. Typed activation
source: `state/replay-model-inspector-head-faq-guide-shadow-r204.json`.
Submission and game/episode selectors are ordered newest ID first; decision
steps retain their chronological order beginning at the earliest recorded
step.

Under revision 205, build and run one exact 1,000-game shadow mirror of the
revision-202 chance-aware inter-turn MCTS design against the identical frozen
revision-195 NO-RTP direct policy. The comparison uses checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
the exact Alakazam pilot deck, and no additional training. It consists of 500
RNG-matched pairs; each pair is played twice with seats swapped, so MCTS is
seat 0 exactly 500 times and seat 1 exactly 500 times. Finish all 1,000 games
rather than stopping early. Parallelize safely across compatible idle remotes,
but do not use or disturb the host/GPU running r198 attempt 10 before its
terminal boundary and never interfere with an interactive session. Enforce
the one typed monotonic-clock budget at exactly 20 seconds of planner work per
actual turn and 5 seconds before an atomic action; partial or unverified trees
time out to the exact direct-policy action. The final report must include
paired and seat-split outcomes, legality/forfeit/failure counts, turn/action
latency distributions, and per-turn search throughput: number of leaf/result
evaluations seen, nodes expanded, cache reuse/rebuild counts, and whether the
requested tree was fully expanded and backed up within budget. Report the
mean, median, p95, minimum, and maximum results seen per turn and full-tree
completion rates overall and by seat. The current phase-1 prebuilt-tree cache
validator is not MCTS and cannot be used as the experimental arm: launch is
fail-closed until a real information-set-safe successor/search path, future
legality and exact simple-chance receipts, hard-clock enforcement, integrity
tests, same-checkpoint parity, remote determinism, and a new content-addressed
evaluation identity all pass. The evaluation is training-ineligible and grants
no serving, production action, selector, publication, Kaggle, promotion,
r175-restart, or iteration-21 authority. Typed evaluation contract:
`state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json`.

Under revision 206, submit exactly two immutable Kaggle A/B variants derived
from the revision-195 terminal Alakazam NO-RTP bundle and its exact checkpoint,
deck, matchup tree, search/belief assets, and turn-order contract. Both remain
explicitly NO RTP. At every factorized legal-action stage, the current-deck
Alakazam guide may add a bounded serving-time bonus to the frozen model's log
policy: copy 1 uses `0.05` and copy 2 uses `0.10`. Guide scores are normalized
within the current legal stage to `[0, 1]`, so those numbers are the maximum
per-option logit nudges and copy 2 is exactly twice copy 1. Missing, malformed,
nonfinite, flat, or tied guide evidence falls back exactly to the model. This
is a deliberately labeled serving experiment, not the historical training
guide-loss multiplier and not a learned guide head. Preserve all earlier
checkpoints and submitted bundles byte-for-byte, use the existing one-shot
Kaggle queue/quota controls, and archive each accepted bundle for replay
reconstruction. Typed source:
`state/alakazam-guide-logit-ab-submissions-r206.json`.

Under revision 207, supersede only the experimental-arm mechanics of the
revision-205 BO1000 shadow mirror. Preserve the exact 1,000-game / 500
RNG-matched seat-swapped-pair design, frozen revision-195 NO-RTP checkpoint
and deck on both arms, training ineligibility, and all existing authority and
hard-clock limits. The experimental arm is simulator-backed chance-aware
inter-turn MCTS: it uses the same checksum-bound frozen model for legal
policy priors and batched frozen outcome/value reranking of nonterminal leaves,
while a simulator terminal leaf records the exact terminal result and may not
be replaced, reweighted, or reranked by a model. No gradients, optimizer
steps, parameter updates, or new training are authorized. The identity-bound
monotonic budgets remain exactly 20 seconds per actual turn and 5 seconds
before each atomic action, charging simulator, prior, reranking, validation,
cache, and backup work. Preserve per-turn and report-level seat splits, and
add separate receipt-backed telemetry for simulator transitions, exact terminal
results, frozen policy priors, and frozen outcome/value leaf reranking. Bert,
Elmo, and train may participate only after a per-host safe-noninterference
preflight; do not disturb attempt 10 or any interactive session. This remains
shadow-only and fail-closed until every simulator, exact-result, frozen-model,
clock, integrity, determinism, noninterference, and content-addressed-output
receipt is valid. It grants no training, serving, action, selector,
publication, Kaggle, promotion, r175-restart, or iteration-21 authority. Typed
contract: `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json`.

Under revision 208, the Replay Model Inspector must distinguish a playground
head's `1.0x` source multiplier from the head's actual nominal baseline Fusion
coefficient. For every eligible active route, show the checksum-bound runtime
components used by the policy: shared total-delta cap, learned route
reliability/multiplier, active-route count, and their nominal coefficient
`cap * multiplier / active_route_count`. This scalar is a pre-nonlinearity
baseline coefficient, not a fixed percentage contribution: each route still
produces option-conditioned signals and the shared mean/tanh fusion makes the
exact policy effect decision-specific. Keep the exact nonlinear recomputation
and leave-one-head-out views as the source of truth for what changing or
removing a head does. This is a read-only inspector presentation/API change;
it grants no training, submission, A/B analysis, or model-mutation authority.
Typed contract:
`state/replay-model-inspector-baseline-head-coefficients-r208.json`.

Under revision 209, add an explicit `Check Kaggle now` control to the Replay
Model Inspector so the owner can request the existing managed submission-replay
sync between hourly timer runs. The button must invoke the same fixed Elmo
oneshot service without changing, disabling, or replacing its hourly timer,
then wait for completion and refresh the read-only inspector index. Keep the
inspector container itself unprivileged and read-only: the authenticated/private
dashboard gateway alone may accept one exact bodyless POST with a fixed custom
intent header and execute one fixed SSH/systemd command. All other inspector
POSTs remain rejected; no request-controlled host, unit, command, arguments, or
paths are allowed. Concurrent requests collapse through systemd's existing
unit state. This grants replay-cache refresh authority only, never training,
submission, model mutation, replay eligibility, or A/B analysis authority.
Typed contract:
`state/replay-model-inspector-manual-replay-sync-r209.json`.

Under revision 210, the owner fully abandons the legacy recursive RTP line and
orders its active r198 attempt-10 evaluation stopped immediately. The managed
service was stopped through systemd without a manual process kill, restart,
reset, cleanup, or deletion. Two concurrent idempotent stop requests crossed
before controller coordination completed; they targeted the same unit and did
not create a second process or restart. Preserve the stopped partial attempt
byte-for-byte: it contains 761 completed transcripts and execution receipts,
253 complete matched cells plus two completed arms of the next cell, no failed
worker evidence, and no terminal evaluator, compiler, HOLD, or promotion
artifact. It is explicitly an incomplete owner-stopped prefix, not a complete
efficacy result, training source, promotion receipt, or action-authority proof.
Do not restart or retry attempt 10, create another legacy recursive-RTP
candidate or evaluation, collect legacy RTP data, attach an RTP sidecar, train
RTP, promote RTP, serve RTP, change the selector for RTP, publish an RTP
checkpoint, or submit RTP to Kaggle. Preserve all r197/r198/r199 source,
candidate, sidecar, output, failure, and evaluation evidence as history. This
abandonment does not cancel the separately versioned revision-202/205/207
simulator-backed MCTS experiment: that strategy must use the exact r195 NO-RTP
model, may not use the abandoned RTP sidecar/executor, and remains authorized
only for offline implementation/preflight and the shadow BO1000 after every
r207 prerequisite passes. It retains zero serving or production action
authority. Typed abandonment contract:
`state/alakazam-rtp-abandonment-r210.json`.

Under revision 211, every newly accepted owner submission whose exact uploaded
bundle, extracted runtime, checkpoint, matchup tree, and replay bytes are
archived must become trace-ready after an automatic submission-specific runtime
attestation; the owner must not have to hand-author a provenance row for each
new model. For a package that includes the revision-206 guide decision policy,
reconstruction must execute that exact package-local post-model rule. The
displayed submitted-runtime probabilities and selected action therefore use
the neural policy plus the packaged normalized guide-logit bonus, while the
unadjusted neural probabilities and action remain separately visible for
comparison. Exact head, adapter, and playground counterfactuals retain the same
package-local guide bonus on both compared paths so their reported effects are
effects within the policy that was actually submitted. A missing or invalid
exact runtime remains fail-closed and may never borrow a neighboring package.
This is read-only causal re-evaluation, not recorded historical logits, model
mutation, training, or A/B outcome analysis. Typed contract:
`state/replay-model-inspector-new-submission-reproduction-r211.json`.

Under revision 212, the owner authorizes one separately managed,
Alakazam-only Guide2Vec distillation and its subsequent no-MCTS BO1000 mirror.
The frozen base on both arms is the exact revision-195 NO-RTP submission
`55378392`, checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
bundle `sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145`,
and r175 Alakazam pilot deck. Train only a 100k--500k parameter
option-conditioned Guide2Vec head from frozen policy-visible hidden state and
the current legal option, never an LLM and never the base model. Its only
teacher target is the existing compact Alakazam guide target index plus
confidence on aligned legal stages; no full teacher-score vector, game action,
outcome, Kaggle replay, Slop Box row, legacy-RTP row, r205/r207 row, or BO1000
row enters training. Train, validation, and heldout partitions must be disjoint
by whole episode and source day, with retained-row, episode, and source-day
fingerprints reported for each partition; deck-list overlap is expected and is
not a split fence. The head may add at most `0.05` normalized logit bonus at
one current legal factorized stage and otherwise falls back exactly to the
frozen direct policy. Start only the dedicated managed Blackwell service
`pokebot-alakazam-guide2vec-r212.service` after a receipt-backed
noninterference preflight, with a separate cgroup, content-addressed source,
and output roots; it must not stop, restart, reconfigure, or reduce another
protected workload or the 96-worker Blackwell contract. Then run exactly 1,000
training-ineligible games as 500 matched RNG/deck-order pairs: each pair has
one candidate-first and one candidate-second game, so Guide2Vec is actual first
exactly 500 times and actual second exactly 500 times, independently verified
from seat. This is neither MCTS nor RTP and may not use r207's runner, search,
simulator leaf reranker, legacy sidecar, executor, or receipts. It remains
shadow-only with no selector, serving, promotion, checkpoint-base mutation,
Kaggle, r175-restart, or iteration-21 authority. Typed contract:
`state/alakazam-guide2vec-no-mcts-bo1000-r212.json`.

Under revision 213, every selected game in the Replay Model Inspector exposes
a presentation-only link to the PTCG Visualizer using exactly the archived
decimal replay/episode ID in
`https://ptcgvis.heroz.jp/Visualizer/Replay/<replay-id>/0`. The link opens in a
new tab and sends no replay payload, model data, dashboard credential, cookie,
or referrer. Invalid or unavailable replay IDs produce no external link. This
adds no external write, training, submission, or replay-mutation authority.
Typed contract:
`state/replay-model-inspector-ptcg-visualizer-link-r213.json`.

Under revision 220, the Replay Model Inspector adds one quick-find input for a
pasted replay URL. The browser extracts only the URL's exact decimal
`submissionId` and `episodeId` query parameters, selects the indexed submission,
waits for its game list, and opens that exact episode. Other query parameters
are irrelevant. The pasted URL is never fetched or navigated to, identifiers
never pass through floating-point conversion, and a missing submission or
episode is reported explicitly instead of opening a different replay. This is
a presentation-only read from the existing inspector index and grants no sync,
submission, replay-mutation, training, or selector authority. Typed contract:
`state/replay-model-inspector-replay-link-quick-find-r220.json`.

Under revision 221, supersede only the stochastic fallback semantics of the
unlaunched revision-219 local multi-search-turn BeliefMCTS mirror; preserve the
revision-219 typed source byte-for-byte and preserve its 45-second shared
per-actual-turn planner pool, 15-second maximum meaningful search segment,
residual-pool re-searches, deterministic cache behavior, 10-game/5-pair canary,
500-pair/1,000-game mirror, frozen r195 NO-RTP model, runtime-on Matchup
Adapter, no-training, no-serving, no-selector, and no-Kaggle boundaries. A
fully exposed finite chance point still may be searched beyond only when every
two-to-six outcome, exact probability, independently forceable successor, and
future legal-action set is available and receipted; force each child from the
same pre-random state, enumerate all outcomes, and back up the exact
probability-weighted value. Paired engine seed material is for match
reproducibility only and may not hunt or pre-randomize a desired chance
outcome. If any one of those proofs is
missing or incomplete, search stops at the *pre-random* boundary and evaluates
that leaf with the frozen model. It must not privately sample a coin, die, or
other outcome; guess game rules, distributions, successors, or future legality;
or advance a simulated branch through an outcome that has not happened in the
real game. No unobserved-outcome cache child is created. Once the real outcome
is observed, normal fingerprint/cache validation applies and the next
meaningful decision may re-search only from the residual shared 45-second turn
pool. A fresh r221 preflight and valid r221 10-game canary are required before
the new content-addressed BO1000; receipts must report exact finite
enumerations, pre-random boundary leaf evaluations and reasons, and zero
private random samples, rule guesses, or unobserved random advances. Typed
contract: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r221.json`.

Under revision 222 (Alakazam MCTS sequencing), supersede only revision 221's
separate-canary launch sequence and its evaluation transport for a fresh
content-addressed local mirror. After fresh r222 preflight, launch one
uninterrupted Blackwell evaluation of all 500 seat-swapped pairs / 1,000 games.
Pairs `0..4` / games `0..9` are a live prefix diagnostic within that already
running job, not a separate canary or validity gate: its result may not pause,
restart, or authorize the remainder. The exact 500/500 MCTS seat and actual
first/second balances are enforced by the wrapper, but the two games in each
seat-swapped pair use independent, unmatched RNG streams; report that truth and
never call the evaluation paired-RNG or seed-matched.

The evaluation must use only the exact stock r195 archived `cg/libcg.so`
(`sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c`,
1,342,400 bytes; `BattleStart`, `SearchBegin`, `SearchStep`, `SearchEnd`). Each
evaluation game gets one fresh OS process and one stock libcg environment.
Private MCTS uses that process's in-process stock Search API; many such game
processes may share Blackwell's queued GPU leaf inference. B77, seeded or
`BattleStartSeeded` engines, batch/multi-game custom engines, and custom force
paths are forbidden. Within every meaningful MCTS decision segment, run exactly
eight concurrent simulation trajectories—not eight games, models, or a literal
beam—against one shared logical MCTS tree and the one frozen r195 model/Adapter
path. The lanes select/reserve work from that same tree; each owns an isolated
stock-libcg Search ID/state. GPU leaf forwards are microbatched across lanes and
every result backs up into that shared tree. A separate root-parallel forest or
root-stat merge, serial lane fallback, and partial-lane MCTS action authority
are forbidden. The r222 preflight hard-fails before launch unless the stock ABI
proves eight simultaneous Search states isolation-safe; it may not
reduce/substitute lanes or share an unisolated state. Report requested/active
lane counts, isolated state count, shared-tree/model integrity, leaf microbatch
count/size distribution, and lane trajectory/backup counts without imputation.
Use virtual-loss/path-and-leaf reservations plus safe in-flight frozen-eval
coalescing/cache to avoid duplicate work; never merge hidden/random/future-
legality worlds from a public lookalike alone. Report dedup/cache hits and
unavoidable repeats, and return every action with zero outstanding
reservations.
The r221 chance rule remains binding: exact enumeration
can happen only if this stock ABI itself demonstrates every required forcing,
outcome, probability, independent-successor, and future-legality proof from the
same pre-random state; it is never assumed. Otherwise search ends at the
pre-random frozen leaf—no private random sample, guessed rule, or unobserved
advance. The same stock one-process Search path must receive a separate
portable-Kaggle compatibility smoke, but that smoke is not a local BO1000 gate
and grants no Kaggle API, queue, upload, or submission authority. All other
r221 45-second/15-second search, frozen r195, Matchup Adapter, no-training,
no-serving, no-selector, and non-promotion constraints remain unchanged. Typed
contract: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json`.

Under revision 225 (Alakazam shared-tree eight-lane diagnostic), preserve r222
byte-for-byte and keep its ordinary 1,000-game Blackwell BO1000 local and
continuous. After an immutable exact-package build and passing local
child-process capability, isolation, cleanup, randomness-boundary, and
throughput receipts, exactly one Kaggle diagnostic is conditionally authorized
for `pokemon-tcg-ai-battle`, labeled exactly `DONT USE FOR REVIEW — 8-LANE
SHARED-TREE VIABILITY`. It hard-fails unless eight simultaneous isolated stock
Search states select/reserve/back up into one shared logical tree with frozen
model leaf microbatching and zero outstanding reservations at action return.
The pre-submit receipt must bind the r225/r222 hashes, candidate archive and
member/entrypoint hashes, stock libcg, frozen model/tree, competition, label,
and eight-lane receipt. No retries, copies, queue/batch action, review,
strength, selector, promotion, or gameplay authority is granted. The expected
resource receipt is AWS p5.4xlarge-equivalent, H100 80 GB, 256 GiB RAM, and 16
vCPU; a mismatch cannot count as a viability pass. r224's distinct root-parallel
2-vCPU/no-GPU candidate remains byte-for-byte untouched. Typed handoff
contract: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 226, abandon the still-unlaunched revision-212 Alakazam
Guide2Vec training and its no-MCTS BO1000 without deleting or rewriting its
sealed artifacts. Replace that one-off launch path with a dormant,
guide-agnostic numeric Guide2Vec training pipeline for a future updated guide.
Keep a guide-independent frozen feature store containing only causal base
`state_vec`, legal-option `option_hidden`, base logits, and ragged option
offsets. Each guide revision supplies a separate checksum-bound label overlay
of target index and confidence joined by exact source/episode/seat/decision/
factorized-stage, policy-visible-observation, and legal-option fingerprints.
When the frozen base, latent ABI, adapter/runtime context, stage index,
extractor, and dtype identities are unchanged, a guide update rematerializes
only its labels and trains a new tiny head; it does not rerun base latent
extraction or retrain Card2Vec/Word2Vec. The pipeline may be implemented and
checked now, but no current guide is ready and no training service, gradient
update, candidate publication, runtime attachment, BO1000, selector, serving,
promotion, Kaggle, RTP, or MCTS change is authorized. A later explicit owner
launch decision plus complete immutable guide, split, alignment, base, host,
source, and output receipts is required. All existing MCTS work and outputs
remain untouched. Typed contract:
`state/guide2vec-general-training-pipeline-r226.json`.

Under revision 227, correct only the active r222/r225 stock-search topology.
For each fresh one-game process there is one Kaggle submission agent/process
and one loaded stock r195 `cg/libcg.so` DSO—not eight competition agents or
eight loaded libraries. That DSO hosts exactly eight internal `AgentStart()`
simulator/search arenas, one persistent arena and CPU worker per lane. Those
internal arenas are implementation-private search contexts rather than
competition agents; each creates one distinct `SearchBegin` ID and may not
concurrently share an `AgentStart` handle. One master owns the sole mutable
logical MCTS tree: it selects/reserves eight paths, gathers their eight frontier
leaves into one frozen r195 GPU batch, backs up all eight results into that one
tree, releases the batch reservations, and repeats until the existing segment
budget ends. The eight-lane batch is latency hiding for CPU-to-GPU simulation,
not a one-lane baseline or ratio experiment; per-lane GPU batches, private
trees/root merges, reduced lanes, partial-lane authority, and a one-lane
baseline/ratio are forbidden. The correction changes no r221 randomness,
frozen r195, timing, authority, r224, r222 BO1000 continuity, or r225 one-shot
semantics. Typed canonical sources:
`state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json` and
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 229-MCTS, Kaggle has confirmed that the r228 asynchronous
eight-worker topology is usable. Supersede only r228's active-deliverable
priority and run a durable 500-pair / 1,000-game mirrored evaluation of the
exact r228 shared-tree MCTS decision layer against the standard no-MCTS r195
policy. Both arms use checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
the same frozen r195 model/package/deck/Matchup Adapter assets, stock libcg,
and every non-search setting; only MCTS action selection differs. Each pair
swaps which seat owns MCTS and all 1,000 games finish without early best-of
stopping. Dispatch through the existing receipt-backed scheduler across Elmo,
Bert, and Train/Inzi, using only capacity proven free by managed-service state
and never disrupting a healthy protected workload or interactive session.
Emit content-addressed game and decision telemetry sufficient to report wins,
losses, draws, paired and seat-split win rates; total and per-game decisions
seen; branching, MCTS-eligible, searched, forced, and fallback decision counts;
mean, median, and distribution summaries; end-to-end and per-host throughput;
and search latency, backups, and batching. For every branching MCTS decision,
record the same-state frozen-direct counterfactual, including both actions,
probabilities/ranks, gap, visits, and value evidence. Report both raw action
changes and meaningful changes; an MCTS invocation alone is never a meaningful
change. Fail closed on identity drift, unsafe host admission, duplicate or
partial games, missing telemetry, or structural MCTS failure. Evaluation data
is permanently training-ineligible and grants no selector, serving, promotion,
checkpoint, Kaggle, RTP, or iteration authority. Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 232-MCTS, correct the premise using the authoritative Kaggle
result: r228 submission `55416396` ended `SubmissionStatus.ERROR` with the
reported description `Hung process — 8-LANE SHARED-TREE VIABILITY`; it is not
a passing Kaggle-usability receipt. Preserve that failed submission as evidence.
The r229 implementation remains active, but BO1000 launch now requires a fresh
hang-containment preflight proving bounded per-decision and per-game wall time,
complete eight-lane cleanup, process exit, immutable failed-attempt receipts,
exact-game requeue, and continued healthy-fleet progress. Never call the rough
r228 package Kaggle-usable or let a hung child retain a scheduler slot forever.
Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 233-MCTS, preserve the Kaggle-specific r228 MCTS/simulator
lifecycle unchanged in this BO1000 implementation. Contain simulator stalls
only at the outer one-game evaluation-process boundary: a hard wall-time
watchdog owns only its exact spawned child/process group, emits a non-success
timeout marker, preserves the immutable attempt log and receipt, and lets the
fleet dispatcher requeue the exact game without credit. Healthy workers keep
running and only the repeatedly failing execution slot is quarantined under
revision 231. This outer containment may not alter an action, convert a partial
game into evidence, touch an interactive session, or restart a managed
training/Kaggle service. Kaggle-specific bounded-search/fallback repair belongs
to its separately coordinated task and is not part of the r229 fleet files.
Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 234-Kaggle, supersede only r228's Kaggle execution boundary.
The authoritative failed evidence is submission `55416396`, episode
`91766923`: it ended `TIMEOUT` after a final approximately `438.994`-second
unreturned callback whose complete ordered root legal set had two options.
The packaged 4,096 root ceiling was therefore not causal for that final root.
A private 6,720-action leaf causing `ActionSpaceTooLarge` followed by a
deterministic consumed-completion cleanup deadlock is plausible, but the exact
leaf and causal sequence remain unproven and must not be reported as fact.

For each Kaggle physical game, the top-level submission parent must first
precompute and validate the exact frozen-r195 direct action against the
complete ordered root legal set and its legal fingerprint. It then fresh-execs
one persistent MCTS child for that physical game. The child owns its own
checksum-identical frozen-r195 model, one stock `cg/libcg.so` mapping, one
logical tree, and exactly eight internal `AgentStart()` search arenas/threads.
This deliberately creates two isolated runtime processes, model loads, and DSO
mappings; only the MCTS child owns the eight search arenas. The parent owns the
actual action/fallback authority and monotonic deadline.

On a child timeout, crash, protocol, evaluator, native, or cleanup fault, the
parent may boundedly terminate and reap only that exact owned child, discard
all child/tree/partial-lane state, return only the already precomputed legal
direct action, disable MCTS for the rest of that game, and emit a degraded
marker. Such a game earns no viability-success credit. An invalid direct
action, root, model, or identity binding, or a child that cannot be reaped,
must exit nonzero rather than fallback. Set the complete ordered action ceiling
to exactly 65,536 at the root and every private leaf: the observed 6,720-action
case is fully enumerated; an over-cap root hard-fails, and an over-cap internal
leaf may use only the contained degraded fallback. Sampling, pruning, or
reinterpreting legal choices is forbidden. Receipts must include per-lane
progress, child PID/start identity, termination/reap evidence, direct-action
legal fingerprint, process/model/DSO identities, and the degraded state.
Only local package, fault-injection, and full-game preflight are authorized.
No new Kaggle upload, retry, queue, or copy is authorized without a separate
owner order. Revision 233's r229 outer watchdog, all BO1000 behavior, and all
BO files remain unchanged. Typed source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 235-Kaggle, authorize exactly one new direct replacement Kaggle
diagnostic only after every revision-234 repair gate passes. The required
gates are a checksum-bound repaired package; focused fault suite; explicit
non-reap and legality hard-fail tests; the exact saved episode `91766923`
seat-0 history replay through its final step-58 two-choice callback; an
exact-package full local game; resource, memory, startup, and throughput
preflight; and an immutable binding receipt. That saved replay is the local
regression for the failed game: the repaired exact package must return a legal
action before the hard deadline, either through valid MCTS or the contained
precomputed direct fallback, and must prove broker-child reap when a fault is
injected. The replacement's Kaggle validation episode will differ and cannot
substitute for this replay.

Only when all gates and the binding pass, one operator-directed direct Kaggle
API/upload is permitted with the unique label `DONT USE FOR REVIEW — R235
BOUNDED MCTS FALLBACK TEST`. Before readiness, API/upload authority remains
false. Queueing, automatic retry, copying, a second upload, or any submission
after a validation failure is forbidden. Preserve and download failure logs,
but do not retry without a new owner order. The replacement remains a
diagnostic only: it is nontraining, nonpromotion, nonselector, and earns no
production or serving authority. Preserve the consumed failed submission
`55416396` / episode `91766923` as immutable evidence. Revision 233's r229
outer watchdog, all BO1000 behavior, and all BO files remain unchanged. Typed
source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 236, the owner declares the official Kaggle Environments
`1.32.6` CABT native-library set the new canonical `libcg` for every newly
built simulator package and preflight.  Its Linux x86-64 `cg/libcg.so` is
`sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7`;
the exact Linux aarch64, macOS arm64, and Windows x86-64 siblings are bound in
the typed source.  The source is the checksum-verified official wheel
`kaggle_environments-1.32.6-py3-none-any.whl` and the upstream July-23 CABT
crash-fix commit.  Every new r235 Kaggle package and r229 BO1000 package must
overlay and checksum-bind the appropriate exact official bytes; the complete
cross-platform set may not mix old and new members, and both r229 arms must use
the same identity on every host.  The frozen r195 model, checkpoint, deck,
matchup tree, and policy remain unchanged.  Historical packages, failed
submission `55416396`, episode `91766923`, and their old `ffd89bf9…` simulator
identity remain immutable evidence and are never relabelled.  Revision 236
supersedes revisions 233–235 only for native simulator-binary identity:
r229 retains its outer one-game process watchdog and must not import the
r234 Kaggle broker/direct-fallback or queue-cleanup lifecycle.  No managed
service restart is authorized merely by recording the new canonical library;
new execution fails closed until platform-loader/export, package, saved-replay,
full-game, and host-identity receipts are reissued against the new bytes.
Typed source: `state/canonical-libcg-r236.json`.

Under revision 237, remove the Replay Model Inspector's revision-222
20-second browser deadline for a selected trace. Keep inference on Elmo's
`cuda:0`; Bert remains only the separately managed read-only dashboard
gateway/tunnel. Every trace-ready selected submission/game/step/stage continues
through its checksum-attested archived runtime and checkpoint forward
reconstruction even when a cold load takes longer than 20 seconds. Elapsed
browser wait alone must never mark that reconstruction unavailable or discard
its eventual response. Preserve selected-trace-only loading, no whole-game
prefetch, exact replay/runtime/checkpoint digest gates, serialized one-model GPU
residency, request-local runtime-state restoration, and
`recomputed_not_historical` truth labels. A browser may still abort a request
made stale by a newer selection, and an owned isolated worker retains bounded
crash/hang cleanup; neither path may substitute another model or make
evaluation replays training-eligible. Typed source:
`state/replay-model-inspector-forward-pass-reconstruction-r237.json`. The
inspector-only activation completed on Elmo at `2026-08-10T23:19:01Z`; both a
140.240407-second cold reconstruction and a 53.823427-second resident-model
reconstruction returned HTTP 200 with verified provenance and
`recomputed_not_historical`. A different runtime package then completed its
checksum-bound isolated forward in 52.160755 seconds with the same verified
truth labels, proving one-model residency is not an eligibility gate.
Activation receipt:
`state/replay-model-inspector-forward-pass-reconstruction-activation-r237.json`.

Under revision 238-Kaggle, supersede the old eight-lane r228/r225 viability
requirement only for the new revision-235 replacement package. The Phase-1
Kaggle submission environment is receipt-bound as **11.8 GiB HDD space**,
**12.2 GiB RAM**, **2 vCPUs**, and a **197.7 MiB submission archive limit**.
The replacement must use exactly two isolated internal simulator/search lanes
on one child-owned shared logical MCTS tree: two distinct internal
`AgentStart()` arenas/handles, two distinct `SearchBegin` IDs, and a
two-frontier-leaf frozen-evaluator batch per round. It must hard-fail MCTS
action authority if the exact two-lane topology is not present; it may not
silently fall back to one lane, retain an eight-lane requirement, sample,
prune, or reinterpret legal actions.

Revision 238 retains the revision-234 bounded exact-child broker and
precomputed direct-policy fallback, the complete ordered 65,536 action cap,
the revision-236 canonical libcg set, every r235 local gate, and the exact
single-upload label `DONT USE FOR REVIEW — R235 BOUNDED MCTS FALLBACK TEST`.
It changes neither the consumed r228 submission `55416396` / episode
`91766923` nor its old simulator identity; those remain immutable historical
evidence. It also does not change r222 or r229: r229 retains the r233 outer
one-game watchdog and continues to exclude the r234 Kaggle broker/direct
fallback and queue-cleanup lifecycle. This is a staged package/preflight
change only—no upload, managed-service restart, training, or BO1000 action is
authorized merely by recording it. Typed source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 239-BO1000, supersede the r229 fleet's former eight-lane
topology only for its next package and evaluation. It must use exactly two
isolated internal simulator/search lanes per r228-MCTS game, with two distinct
arenas/handles and `SearchBegin` IDs selecting, reserving, evaluating, and
backing up into its one shared logical tree. The sealed eight-lane r236 package
is preflight-only historical evidence, is not eligible to execute a game, and
started no r229 game. The new two-lane package requires fresh content-addressed
source/package/preflight and receipt binding before any fleet execution.

Revision 239 preserves r230's complete ordered 65,536-action cap, r233's
outer one-game watchdog/requeue/quarantine boundary, and r236's exact official
per-platform libcg set. It deliberately retains r229's pre-r234 Kaggle
lifecycle baseline: it must not import the r234 Kaggle parent broker,
precomputed-direct fallback, or queue-cleanup lifecycle. The topology change
does not authorize managed-service restart, training, serving, selector,
Kaggle, or premature fleet execution; no script/config/test mutation is
implied by this record. Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 240-Kaggle, supersede only the r235 replacement package's
decision scheduling. The parent still precomputes and validates the complete,
legal frozen-r195 direct action and legal fingerprint first. When its selected
frozen-r195 factorized probability is finite and at least `0.90` at **every**
selected factorized stage, the parent returns that exact legal action
immediately in mode `high_confidence_frozen_direct`; it must not start or call
the MCTS child, journals the stage probabilities and threshold decision, and
is not degraded. Missing, malformed, non-finite, or below-threshold confidence
at any selected stage is ambiguous and goes to MCTS rather than silently
qualifying for this direct path.

Ambiguous MCTS remains exactly two lanes and is bounded by a 2.0-second hard
child search budget and a 4.0-second parent action deadline. It may early-stop
only after at least 8 completed backups, the same deterministic root leader is
observed three times, and both lanes have progressed; it must stop at 32
backups. A zero-backup result returns only the already precomputed legal direct
action under the existing clean-deadline/containment rules. Revision 240
retains r234 exact-child containment, r236 libcg, the 65,536 complete-action
cap, r238 Phase-1 resource envelope, and the exact one-shot R235 label/upload
boundary. The old r228 fixed eight-second branching windows remain historical
evidence, not a current budget; r229/BO1000 is unchanged. Typed source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

An ambiguous-MCTS receipt may additionally carry a deterministic continuation
plan of at most eight actions. The parent always computes the exact direct
answer first, but uses the next planned action without a new search only when
the current canonical observation fingerprint exactly matches, we remain the
actor, the action is in the current complete legal order, the plan extraction
proved both lanes saw that same fingerprint and agreed on a backed leader, and
no chance/boundary/opponent transition occurred. A plan mismatch, hidden lane
disagreement, randomness, boundary/opponent transition, or illegal action
clears the entire plan and returns to the normal high-confidence-direct or
adaptive-MCTS route. The parent rewrites history to the actual planned action,
journals exactly once, and logs planned-versus-direct; a valid continuation
plan therefore has precedence over the normal high-confidence routing only at
its proven deterministic continuation step.

Under revision 241-TRAINING, start a separately versioned, direct-policy-only
Alakazam successor from the immutable revision-195 terminal expert checkpoint
using the owner's exact new 60-card list and supplied pilot guide. Run exactly
ten RL updates (`iter_00000` through `iter_00009`), with exactly 1,024
self-play games and at least 1,024 games against the checksum-pinned H10 Marnie
direct policy in every update. Preserve the established 8,196-game loop and
exact 50/50 training-seat split. MCTS, RTP, Guide2Vec, search targets, hidden
snapshot labels, and search/belief package assets have no training, opponent,
or submission authority in this lineage.

Use the official Kaggle Environments 1.32.6 revision-236 `libcg` through a
sealed `CG_LIB_PATH`; the private hidden or batch library environment routes
must be absent. Use the strict rolling 20-calendar-day expert replay window
ending today, exactly `2026-07-22..2026-08-10`. Run a five-epoch expert soft
refresh after completed updates five and ten. The update-ten refresh produces
`expert_before_iter_00010.pt` after `iter_00009` and must not trigger an
eleventh collection. Submit exactly once, `first_if_allowed`, only from that
terminal refreshed checkpoint; no update-five, retry, copy, or duplicate
submission is allowed. The exact July 22--August 10 archive is bound by its
immutable receipt; no fallback date or unreceipted start is allowed. Typed source:
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 245-TRAINING, clarify revision 241 without broadening its
changes: preserve the peak revision-195 Alakazam model and loop behavior while
changing only the owner's exact deck, supplied guide, official revision-236
simulator binding, ten-update horizon, and requested expert-refresh schedule.
Every architecture-present non-combo learner head and its bounded fusion route
remain live and trainable; `combo_state` remains present but its loss and
fusion route stay off exactly as in revision 195. Matchup Adapters remain on
for training and the terminal direct-policy submission, with the exact trained
bank and checksum-bound public matchup tree preserved. The established 7,172-
game public mix remains diverse and unchanged except for enforcing the existing
minimum of 1,024 direct H10 Marnie games; H10 Marnie is not the sole public
opponent. The current-deck guide remains training-only and may not suppress,
replace, or disable any learned head, fusion route, Matchup Adapter, public-mix
row, or research-control phase. Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 247-TRAINING, refresh the staged revision-241 Matchup Adapter
archetype roster from the newest completed authenticated PTCGReplay ingest at
`https://ptcgreplay.netlify.app/`.  The authenticated source may define exact
numeric archetype identities, exact names, prevalence, and priority only; it
does not supply actions, gradients, hidden state, outcomes, or gate evidence.
Preserve every existing Router Format 6 slot identity and every peak-r195
adapter tensor byte-for-byte.  Allocate each newly verified archetype only to
the lowest never-used slot, never recycle or reindex a slot, and initialize a
new route exact-zero/dormant.  A new route may train and become runtime-active
only after its own checksum-bound replay support, causal-router fit, precision,
support, zero-state bypass, and activation receipts pass.  Do not mutate or
restart the already-running exact-20-day roster-18 corpus jobs merely to add
the identities; fold eligible checksum-backed data through the ordinary r241
adapter/rehearsal path without changing the exact ten-update, 1,024 + 7,172,
five-epoch refresh, exact deck/guide, direct-policy-only, or one-terminal-submit
contracts.  The site credential is runtime-only and must never be committed,
logged, or written into a receipt.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`; the global slot registry is
updated only after the authenticated snapshot is sealed.

Under revision 248-TRAINING, defer revision 247's PTCGReplay archetype refresh
for now and remove it from every revision-241 launch, training, terminal, and
submission gate.  Do not authenticate, ingest, allocate, fit, or activate a
new matchup archetype in this cycle.  Keep the global Router Format 6 registry
and all existing slot identities unchanged, preserve the peak-r195 E60 public
matchup tree and adapter tensors, and require the r241 checkpoint-derived
adapter migration audit to report `no_slot_change`.  The generic append-only
validator and Chrome installation may remain inert preparation for a later
explicit owner request, but they have no source, training, runtime, selector,
package, or submission authority now.  This deferral changes none of r241's
exact deck, supplied guide, direct-policy-only simulator, 1,024 + 7,172
collection, ten-update, five-epoch refresh, full-head, adapter-runtime-on, or
one-terminal-submit requirements.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 269-TRAINING, supersede revision 248's deferral before r241
starts.  Use Elmo to collect and seal the newest completed authenticated
PTCGReplay ingest from `https://ptcgreplay.netlify.app/`, then prepare the new
Matchup Adapter identities and routes as an explicit pre-start readiness gate.
The source may supply only exact numeric archetype identity, exact name,
prevalence, and priority; it supplies no actions, gradients, hidden state,
outcomes, training rows, or gate wins.  Preserve Router Format 6 slots 0--19
and every inherited r195 adapter tensor byte-for-byte.  Allocate newly verified
archetypes monotonically into the lowest never-used slots beginning at 20;
never rename, reindex, retire, or reuse an existing slot.  Every new slot starts
exact-zero and dormant, and becomes training-ready only after checksum-bound
source snapshot, replay-support, causal-router fit, precision/support,
zero-state bypass, seat/disjointness, and immutable adapter-readiness receipts.
Runtime activation remains separately receipt-gated.  The exact frozen Kaggle
submission 55378392 research opponent receives a distinct immutable opponent
identity/route and may not be aliased to the Alakazam learner route or supply
weights, RTP, traces, or targets.  Collection and fitting run only as new
managed Elmo workloads and must not restart, reconfigure, preempt, or share
mutable outputs with the healthy r259 producer.  Bind the sealed roster,
snapshot, slot allocation, and ready adapter artifacts into the one r241
activation overlay before training starts.  Preserve revisions 263--268's
exact deck and guide, 22 active heads/routes, tactical ablations, 25-update
cycle, fixed 1,024 + 7,172 game mix, r195 research counts, and six submissions.
The PTCGReplay credential remains runtime-only and may never be committed,
logged, or included in a receipt.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 270-TRAINING, collect the decklists exposed by the completed
PTCGReplay Meta snapshot alongside its archetype rows, but treat them only as
guide/reference material for adapter preparation.  Their card multisets and
core-card summaries may support soft causal card-signature features, replay
discovery, confidence diagnostics, and human review.  They are not exact deck
requirements, hard signature rules, action or outcome labels, eligibility
conditions, routing proofs, gate evidence, or reasons to reject a valid
observed variant.  Training support and activation still require independent
checksum-backed replay observations and the revision-269 causal-router,
precision/support, bypass, disjointness, and readiness receipts.  Typed source
remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 271-TRAINING, train a separately versioned causal Matchup
Adapter router candidate on Elmo for the revision-269/270 expanded Meta
snapshot roster before r241 starts.  The existing router, crosswalk, slots
0--19, and r195 adapter bank remain immutable and selectable history.  Fit only
from checksum-backed public-state replay observations and source-disjoint
splits; PTCGReplay names, prevalence, core-card summaries, and guide decklists
may identify/stratify examples but do not become action labels, hard deck
rules, or proof.  Require per-identity and aggregate held-out precision,
weighted support, ambiguity/unknown bypass, seat balance, calibration,
collision, latency, deterministic parity, and exact slot-crosswalk receipts.
The frozen submission-55378392 research opponent retains its distinct package
identity and route and may never alias the Alakazam learner identity.  Publish
the candidate and its fit receipts as separate managed Elmo artifacts; do not
touch r259 or an interactive session.  Select or activate the new router only
through the one pre-start activation overlay after all gates pass.  If it
fails, preserve the failure and keep training blocked rather than weakening a
threshold or silently falling back to an incomplete expanded roster.  Typed
source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 272-TRAINING, append every real numeric archetype identity in
the completed revision-269 Meta snapshot window to the Format-6 roster, not
only the top-20 display subset.  Exclude the pooled `Rogue / Other` row because
the source explicitly defines it as a non-archetype bucket.  Preserve every
existing identity and allocate each unseen numeric source ID exactly once in
stable snapshot order to the next never-used slot.  An appended identity may
remain exact-zero/dormant when replay support is insufficient; inclusion in
the roster does not waive revision-269/271 training-readiness or runtime-
activation gates.  Keep the frozen submission-55378392 opponent identity
distinct from all source archetypes and from the Alakazam learner route.  Typed
source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 273-TRAINING, key every collected Meta snapshot row, guide
decklist, core-card summary, replay-support index, router target, crosswalk
entry, and adapter allocation by its exact PTCGReplay numeric `source_id`.
Display names are non-authoritative labels only and may not merge, redirect, or
alias identities.  In particular, current source ID 167 remains distinct from
historical Teal Mask Ogerpon ex source ID 151, which stays preserved in its
existing slot.  The frozen Kaggle submission 55378392 remains in a separate
package-identity namespace and is never coerced into a PTCGReplay source ID.
Typed source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 274-TRAINING, treat the already sealed twenty-of-twenty Inzi
daily-shard transfer as the payload handoff boundary.  Perform the remaining
deterministic join, schema/count/digest validation, causal parity, completion,
aggregate binding, pre-start canary preparation, activation-overlay
publication, and managed trainer setup locally on Inzi.  Do not retransmit the
daily shard payloads or require a second Elmo-built copy of the same joined
dataset.  Elmo remains read-only source and immutable receipt truth and may
provide only compact source identities, receipts, and adapter artifacts needed
by the Inzi gates.  Rehash every Inzi daily file against its immutable per-day
transfer evidence, ignore dot-prefixed failed-transfer remnants, preserve all
original per-day receipts, and issue a new revision-274 local-post-transfer
attestation.  The final canonical root remains training-ineligible until the
complete local join/promotion/binding/parity and every revision-263--273
all-head, tactical, sidecar, adapter, canary, and overlay gate passes.  Inzi is
the sole training host and no service may start from partial or stale evidence.
After setup and run artifacts are complete, a low-priority create-only return
copy to Elmo is permitted for archival replication; it is not a launch or
training gate and may not interrupt training.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 275-TRAINING, convert the never-started new Alakazam training
lineage from runtime candidate revision 241 to runtime candidate revision 274.
The active candidate ID, run root, managed trainer/finalizer/queue units,
runtime registry, activation overlay, start authorization, receipts, packages,
and dashboard labels must all use revision 274.  Revision 241 remains
immutable design and transfer provenance only; it has no trainer, selector,
submission, or serving authority.  Preserve the already transferred r260
dataset at its checksum-bound physical staging path rather than copying the
large payload merely to rename a directory; its revision-274 binding must name
it explicitly as imported r241/r260 provenance.  The full revision-263--273
OwnDeckLedger, tactical-outlook, all-22-head/route, exact-ID Matchup Adapter,
25-update, and six-submission contract is unchanged.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json` as the historical gateway for
this converted candidate.

Under revision 276-TRAINING, accelerate the revision-274 simulator collection
with checksum-gated `LibcgMultiEnv` packing.  Use four environments per pack
for self-play and for exact public packages whose opponent ID, content digest,
portable-baseline contract, training group, worker capability, and retention
receipt match the immutable revision-182 safety allowlist.  Unknown, changed,
legacy, malformed, research-package, or otherwise unverified public opponents
remain singleton `play` jobs; in particular, the exact submission 55378392
research cell remains singleton unless it later receives its own separate
retention attestation.  Keep remote farms engaged while using packed transport,
retain independent child identity/result/accounting, and use the available
Inzi simulator cores without changing the exact 1,024 self-play + 7,172 public
= 8,196 games, 4,098/4,098 seats, ≥1,024 H10 Marnie, or ≥128 exact r195
floors.  Multi-env failure falls back only to the verified singleton path and
may not reduce counts, weaken gates, introduce RTP/search, or change policy
semantics.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 277-TRAINING, keep the new four-output tactical-sequence
outcome head in shadow-training mode throughout the exact 25-epoch expert
bootstrap.  Its masked causal loss remains `0.025`, but its Fusion route must
contribute exactly zero to bootstrap policy logits and cannot influence the
bootstrap submission.  After the bootstrap and its submission are both
checksum-receipted, activate the learned zero-safe route before RL update 0;
it then participates in all 25 RL updates and later five-epoch refreshes while
the bounded tactical search planner itself remains shadow-only and has no
dispatch, selector, serving, package, or submission authority.  The activation
requires finite-gradient, label-coverage, calibration, bounded-influence,
direct-action-parity, and route-on/off impact receipts.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 278-KAGGLE, submit exactly one testing-only raw derivative of
the immutable revision-195 NO-RTP Alakazam submission `55378392` using the
owner's exact new revision-241 60-card Alakazam list.  Replace only the
`deck.csv` member; preserve the r195 model, runtime, Matchup Adapter tree,
NO-RTP profile, search-disabled assets, and every other archive member
byte-for-byte.  Perform no training, retraining, checkpoint mutation,
registration, selector change, promotion, or training-service interruption.
Use `first_if_allowed` and the exact visible Kaggle label
`r195 NO RTP raw resubmit, new Alakazam list, not retrained otherwise`.
This is one-shot test evidence only and does not consume, replace, retry, or
change any revision-274 training or six-submission obligation.  This completed
as Kaggle submission `55437592` at `2026-08-11T16:30:18Z`; evaluation was
pending at activation.  The derivative bundle is
`sha256:e5f89d16afcd68c20a69b85a134138adaf5cf514658224b740e829141a7cca27`.
Typed source: `state/alakazam-r195-new-list-no-retrain-r278.json`.

Under revision 279-TRAINING, stop and permanently supersede the r274
resident-Python-object bootstrap attempt before it reaches GPU training.  Its
managed service was stopped cleanly through systemd; preserve its journal and
resource evidence, but never restart that loader or claim learned progress from
it.  Before restarting the 25-epoch bootstrap, build exactly one reusable,
checksum-bound joined training pack locally on Inzi from the already transferred
twenty Alakazam daily feature shards and OwnDeck side store.  Join OwnDeckLedger
by one bulk key operation rather than per-decision SQLite point queries, attach
the tactical overlay exactly once, and encode the complete learner corpus as
flat contiguous numeric arrays with explicit variable-length offsets.  The pack
must retain every applicable ordinary, OwnDeck, visible-tutor,
terminal-conversion, combo, and shadow-tactical target and mask; it may not
fall back to resident Python game/decision objects for epoch training.

Fail closed unless the sealed pack validates exactly 26,704 Alakazam
acting-seat games and 2,040,911 Alakazam decisions, the exact twenty-day source
and feature digests, all sidecar/tactical provenance, offset bounds, tensor
dtypes/shapes, selected-option legality, source-disjoint split identity, and
the r274 all-22-head/bootstrap-route contract.  Train from pinned CPU-memory
batches streamed to `cuda:1`; build/join occurs once and all 25 epochs plus the
later five-epoch refreshes reuse the same immutable pack.  This changes only
data representation and loading: the r195 parent, exact deck, Matchup Adapter
gates, shadow tactical route boundary, six-submission schedule, 25 RL updates,
direct-policy/RTP-off behavior, and every revision-263--277 requirement remain
unchanged.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 280-TRAINING, keep the immutable revision-279 contiguous pack
as the restart-safe CPU artifact, but make full numeric GPU residency on
Inzi's `cuda:1` RTX PRO 5000 Blackwell the primary epoch-training path.  The
sealed pack is 5,725,073,070 bytes and the measured device capacity is 48,935
MiB, so copy every corpus and side tensor to `cuda:1` once after checksum and
shape validation, then derive each game batch by device-side index gathering.
Do not rebuild Python game or decision objects, perform host-side row decoding,
or stream the same batch from CPU during ordinary epoch training.  Preserve
the CPU pack as immutable restart/cache truth and retain pinned CPU-to-GPU
batch streaming only as an explicit fail-closed fallback after a measured GPU
allocation or safety-headroom failure; never fall back to the superseded
resident-object loader.  This data-residency correction changes no parent,
target, mask, loss, head, route, adapter, split, epoch, update, simulator,
RTP, deck, submission, or activation-gate contract.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 281-TRAINING, correct the completed r280 bootstrap boundary:
the ordinary 25-epoch expert pass is not a complete r274 bootstrap when its
receipt reports zero Matchup Adapter rows or leaves the eligible adapter bank
unchanged.  Before any bootstrap submission, tactical-route activation, or RL
update 0, train the eligible Matchup Adapter bank for the bootstrap from
checksum-backed, training-eligible public-state replay support with exact
archetype/package identities and source-disjoint validation.  This is real
optimizer training, not roster rebinding, router-only fitting, zero-slot
materialization, or preservation evidence.  Emit an immutable derivative
checkpoint and receipt proving nonzero labeled adapter rows, optimizer steps,
finite changed eligible adapter tensors, per-route support/validation, exact
isolation of every non-adapter tensor, and exact-zero dormant behavior for any
slot that lacks all readiness gates.  The already completed ordinary and
tactical optimizer work remains immutable parent evidence and must not be
replayed merely to add the missing adapter phase.  The bootstrap submission
must descend from this adapter-trained child.  Later RL updates retain their
separate one-epoch adapter-only continuation after every update.  This changes
no deck, RTP-off/direct-policy boundary, 25-update/six-submission schedule,
router precision gate, or tactical planner authority.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 282-TRAINING, remove the per-RL-update tactical shadow-search
materialization gate from the active revision-274 25-update loop.  The owner
prefers immediate full-model RL backprop over waiting for 1,024 newly searched
tactical roots on every sealed update.  Preserve the completed expert tactical
bootstrap, its activated learned route, checkpoint, receipts, and shadow-only
planner evidence, but do not run `materialize_tactical_shard_overlay` on RL
shards and set the RL tactical-sequence-specific supervised loss to exactly
zero.  The tactical planner retains no dispatch, serving, selector, package, or
submission authority.  Resume update 0 from its immutable exact 8,196-game
collection receipt; do not recollect it.  All ordinary full-model RL losses,
the other 21 heads/routes, one adapter-only continuation epoch, direct-policy
and RTP-off boundaries, exact opponent/seat contracts, 25 updates, five-update
expert refresh cadence, and six-submission schedule remain unchanged.  This is
an immediate pre-gradient boundary change: preserve the superseded partial
materialization process evidence, stop/restart only the declared managed
training service, and proceed to optimizer training.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 283-TRAINING, strengthen revision 282 by physically removing
only the new four-output `tactical_sequence_outcome_head` and its dedicated
`tactical_sequence_outcome_route` from the active r274 learner before update-0
backprop.  Preserve the older strategic `tactical_outcome` head and its Fusion
route unchanged and active.  Create a checksum-bound tactical-free architecture
child in memory from the immutable activated bootstrap parent: the two new
tactical-sequence parameter prefixes must be absent from the update-0
checkpoint, their three model-config gates must be false, and every retained
parent tensor must load unchanged before RL optimization.  The old bootstrap,
route-activation, and partial per-update materialization artifacts remain
immutable historical evidence and are not active lineage requirements.  The
active learner now has 21 heads/routes; all 21 remain active and train normally,
including the old strategic tactical head.  Resume from the sealed update-0
collection without recollection.  Every other revision-282 boundary remains
unchanged.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 284-TRAINING, stop the incomplete revision-192 broad formal
holdout that followed update 0 and preserve its exact completed prefix as
diagnostic-only, training-ineligible, gate-ineligible evidence.  It is neither
a pass nor a completed formal evaluation.  Advance the already trained and
promotion-passed update-0 candidate through an immutable owner-deferral receipt
without recollection or retraining, and run the replacement formal holdout for
the first time after update 1.  Every revision-274 formal holdout from that
point forward contains exactly two checksum-bound opponents and no others: the
frozen revision-195 NO-RTP direct-policy submission `55378392` and the exact
H10 Marnie's Grimmsnarl ex specialist
`specialist-marnie-final-format-h10-f20efb20f5c3`.  Evaluate exactly 250 greedy
games per opponent, split 125/125 by learner seat, for 500 total games with no
early stop.  This formal-only roster change does not narrow the diverse public
training mix, its H10 Marnie and r195 minimum cells, or the separate diagnostic
research-control roster.

At the same update boundary, deactivate `combo_state` again for iteration 1
and every later update and expert refresh: retain the physical historical head
and route tensors for exact checkpoint compatibility, but set its supervised
loss to exactly zero and its Fusion route off.  Do not delete or rewrite its
update-0 learned weights.  The old strategic `tactical_outcome` head remains
active, while the already removed new tactical-sequence head remains absent.
The active learner therefore has 21 physical retained heads, 20 active Fusion
routes, and no combo gradient or policy-logit influence after the boundary.
Bind the holdout deferral, replacement contract, combo shutdown, candidate
digest, partial-prefix evidence, and remote-fleet continuation to one immutable
receipt before restarting only the managed r274 trainer.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 285-TRAINING, shorten only the active r274 RL horizon from 25
updates to exactly 20 updates: commit `iter_00000` through `iter_00019`, finish
with the existing five-epoch refresh and one-shot submission at boundary 20,
and never collect `iter_00020`.  The bootstrap submission plus boundaries 5,
10, 15, and 20 make exactly five authorized submissions for this lineage.
Remove the former boundary-25 refresh/submission obligation.  Preserve every
per-update game, seat, public-mix, adapter, direct-policy/RTP-off, revised
holdout, combo-off, and receipt requirement.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 286-TRAINING, immediately quarantine Elmo from the active r274
production collection because its 36 advertised workers are producing only
about 0.17 games/second and its four-game jobs have emitted repeated
`poke_bot.submission_budget` import failures.  Stop only the managed r274
trainer cleanly, seal the exact iteration-1 shard as an immutable append-only
partial-resume sidecar, and restart that same iteration with Bert as the sole
remote endpoint.  Preserve every completed source game and recollect only
missing job identities; do not change the checkpoint, seeds, schedule, seats,
public mix, formal holdout, adapter learning, direct-policy/RTP-off behavior,
or 20-update horizon.  Keep Elmo's worker managed but outside production while
it is repaired and tested independently.  Elmo may rejoin only after an exact
singleton and four-game package canary proves the complete worker/baseline
import surface, zero recurrence across every worker child, checksum parity,
and materially useful measured throughput, followed by explicit owner
readmission.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 288-TRAINING, add a separately versioned, default-off
Alakazam turn-checklist heuristic logit layer for the exact new-list runtime.
It must expose and score exactly these eight causal, per-turn questions:
`ko_hand_threshold`, `safe_spend_above_threshold`,
`replacement_alakazam_line`, `unavoidable_draws_before_attack`,
`bench_prize_exposure`, `immediate_disruption_outcome`,
`unknown_prize_robust_line`, and `terminal_before_forced_draw`.  Bind the
owner-supplied checklist guide attachment
`sha256:682f60fc9211e8092c7122b2addfcbe0bfe91ca33d75bddf2171007894585cd2`.
The immutable r195 NO-RTP checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`
remains the weight parent, but it is not the exact new-list deck: the runtime
must bind the exact-list canonical multiset
`sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182`.

This is not a guide takeover or a checkpoint-tensor change.  When later
activated, it is a bounded, deterministic residual applied after the current
neural heads, learned Fusion, OwnDeck, and Matchup Adapter effects and before
legal action selection, with total per-option influence clipped to
`[-0.10, +0.10]`.  Historic guide runtime action authority remains false; the
new layer is separately scoped and must be neutral (zero residual with an
unavailable reason) for malformed, unknown, hidden, or otherwise noncausal
evidence.  It must not infer hidden information or invoke search, rollouts,
MCTS, RTP, or a new action authority.

Elmo may run a small, isolated calibration only: freeze every neural
checkpoint tensor and train at most the eight scalar checklist gates plus one
separate guide gate on source-disjoint exact-new-list data.  That calibration
is validation-only; it has no production-loop, learner, selector, submission,
or Elmo-readmission authority.  Do not alter the current active iteration or
its sealed prefix.  Implement and validate in parallel, then activate only at
the next safe receipt-backed boundary.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 289-TRAINING, permit one Elmo-only, training-ineligible
turn-checklist diagnostic BO250 after the r288 implementation and parity
preflight pass.  It is exactly 125 seed-matched, seat-swapped pairs (250 total
games) using the exact new-list deck on both arms.  The candidate arm is the
current receipt-bound r274 direct policy with the separate checklist layer;
the control is the immutable r195 NO-RTP direct policy.  The candidate must
occupy exactly 125 games in each seat and exactly 125 actual-first and 125
actual-second games.  Both arms remain direct policy only: no RTP, search,
MCTS, rollout, hidden inference, calibration leakage, production selector,
Kaggle, promotion, or learner authority.

Calibration rows and seeds must be source-disjoint from this BO250.  Elmo
remains excluded from production; this diagnostic cannot satisfy, weaken, or
replace any production readmission condition.  Fail closed if the exact
receipt-bound r274 candidate artifact is unavailable or fails parity—never
substitute another checkpoint, direct policy, deck, or layer state.  Do not
touch the active iteration; run only as an isolated receipt-backed diagnostic.
Typed source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 290-TRAINING, make the r288 turn-checklist layer part of the
current active Inzi r274 learner lineage, not a standalone r195-only side
model.  Activate it at the first safe immutable iteration boundary after the
currently in-flight collection is sealed and before the next collection
dispatch.  Never switch it on mid-shard or during the current iteration, and
never recollect, restart, or otherwise disturb the sealed work to install it.
Once receipt-activated, apply the same bounded layer consistently to subsequent
r274 behavior-policy collection, ordinary direct-policy evaluation, refresh
evaluation, and every later receipt-built submission runtime.  Neural
checkpoint/head, OwnDeck, and Matchup Adapter training continue unchanged; the
residual itself is not a trainable checkpoint tensor.  The immutable r195
model remains a parent/control only.

Elmo calibration and the r289 BO250 remain isolated from Inzi and cannot
directly mutate it.  A calibrated configuration is admissible only through a
checksum-bound boundary receipt with parity evidence; it cannot create
production Elmo authority.  RTP, search, and MCTS remain off.  Typed source
remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 291-TRAINING, supersede only revision 290's activation timing.
Do not deploy, activate, or stage an Inzi runtime change for the checklist at
the next boundary.  It remains an intended additive r274-lineage candidate,
but testing is now Elmo-only in an isolated non-production directory and
process.  Elmo may receive create-only copies or read-only pulls of the exact
receipt-bound r274 checkpoint/package/config and immutable r195 control; it
must never modify Inzi files, services, runtime, boundary configuration, or
drop-ins, and must not stop, restart, or preempt the active trainer.  No
production worker change is authorized.

Complete local and Elmo unit, parity, and calibration checks plus the isolated
r289 BO250 first.  Any later Inzi activation requires a new explicit owner
decision after those results; a receipt or boundary alone has no activation
authority.  Elmo remains excluded from production, and RTP/search/MCTS remain
off.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 292-TRAINING, supersede only the semantic interpretation of
turn-checklist channel `replacement_alakazam_line`.  It means the backup
attacker line on the **Bench only**; the current Active Alakazam line is always
excluded.  The per-turn trace must classify that bench-only backup as `ready`
only for a Benched Alakazam with Psychic-providing Energy, `completable` only
for a Benched Kadabra or Abra with every exact, visible evolution, Energy, and
timing resource needed, and `not_live` when it depends on an unknown draw or
any possibly prized key card.  Unknown/malformed evolution eligibility or
timing is `unavailable`, neutral, and must never be guessed live.  A regression
must prove that an Active Alakazam alone does not count as a backup line or
produce positive `replacement_alakazam_line` evidence.

Preserve revision 291's Elmo-only non-production boundary: do not touch Inzi,
its services, runtime, boundary configuration, or active trainer.  The r289
BO250 remains delayed until the corrected bench-only regression, local/Elmo
unit checks, and parity checks all pass; it remains training-ineligible and
does not create any production or readmission authority.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 293-TRAINING, audit the turn-checklist layer against every
existing active r274 learned surface—current neural heads, learned Fusion,
OwnDeck, and Matchup Adapter effects—but do not edit any of those existing
logics by default.  Resolve overlap or double counting only through the new
checklist layer's gates and per-channel trace: the trace must identify the
existing-route overlap or distinct rationale, the new-layer attenuation or
suppression decision, and its post-deduplication signed residual.  An existing
head, Fusion route, OwnDeck path, or Matchup Adapter may change only after a
drastic correctness issue is separately evidenced and reported before action.

The legacy broad guide scorer has no runtime residual authority while known
semantic contradictions remain.  Validated rules derived from the owner
attachment may inform the eight named checklist channels, but the separate
guide gate is default/exact zero or trace-only until source-disjoint corrected
validation resolves those contradictions.  Preserve revision 291's Elmo-only
non-production / no-Inzi boundary.  The r289 BO250 is delayed until this
overlap audit and the guide-gate trace-or-zero receipt pass, alongside the
existing r292 checks.  This record authorizes no code or service change. Typed
source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 294-TRAINING, bind the owner-supplied guide attachment
`/Users/tsinzitari/.codex/attachments/fdee1eae-e3b4-4631-bb52-864f8f7e68d9/pasted-text.txt`
(`sha256:4580196717277a5d5672eb44bc5c69c9de56ca9ebb9ca93fa76a2c9b5ba278a3`)
as the superseding source for `poke_bot/alakazam_new_list_heuristics.py`
scorer semantics.  That existing scorer must be updated from the new guide,
while learned neural heads, Fusion, OwnDeck, and Matchup Adapter logic remain
unchanged.  The attachment's then-incomplete four-Alakazam inventory sentence
did not provide a complete replacement 60-card list and conflicted with the
checksum-bound three-copy canonical list, so it had no deck authority and did
not alter the exact list or multiset.  The guide scorer remains training-only
with no direct runtime action authority.  Validated deterministic attachment
rules may inform the eight checklist channels, but broad `guide_support`
remains exact-zero or trace-only until corrected source-disjoint validation.
Preserve the Elmo-only/no-Inzi boundary.  This record performs no code or
service change.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 295-TRAINING, supersede only revision 294's attachment byte
identity because that same attachment path was corrected in place.  Its new
identity is
`sha256:5cc092c9ed93b3e0e4ecae9fca9d50409bea6979e8d92e358f684091e0cdff8b`.
The owner-supplied corrected guide now supplies the exact 17-Pokémon,
36-Trainer, 7-Energy inventory, including three Alakazam, and it matches the
existing checksum-bound canonical 60-card list and multiset.  Preserve those
canonical deck bytes and digest unchanged.  The required scorer update remains
training-only with no direct runtime action authority; learned neural heads,
Fusion, OwnDeck, and Matchup Adapter logic remain immutable under this review.
Checklist channels may use validated deterministic guide rules, while broad
`guide_support` stays exact-zero or trace-only until source-disjoint corrected
validation.  Preserve Elmo-only/no-Inzi isolation.  This record performs no
code or service change.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 296-TRAINING, clarify the multi-environment scheduler's worker
accounting for every remote that is later explicitly readmitted.  An
endpoint's advertised worker count is concurrent worker-process/socket
capacity; a four-game `self_play_multi` packet is four environments executed
behind one such worker.  Effective environment capacity is advertised workers
multiplied by four, never advertised workers divided by four and never a
packed packet collapsed into one source game.  Every child retains its
independent job, seed, seat, result, trajectory, and receipt identity.  This
does not readmit Elmo or Bert, alter the current Inzi-only production
override, or weaken any health and explicit-owner readmission requirement.
RTP/search/MCTS remain off.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 297-TRAINING, the active Inzi production collector has a hard
topology of exactly 32 simulator processes with four `LibcgMultiEnv`
environments per process (128 concurrent environments).  Short startup-window
GPS measurements do not authorize reducing that concurrency or replacing the
owner's sustained measurement.  Keep the corrected dual-GPU routing, the
Inzi-only/no-remotes override, append-only partial-shard preservation, and
RTP/search/MCTS-off behavior.  Any later remote readmission inherits revision
296's worker-times-four accounting.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 299-TRAINING, the owner rescinds revision 297's hard requirement
that active Inzi production remain at exactly 32 simulator processes. Preserve
four `LibcgMultiEnv` environments per selected simulator process, but choose the
next process count from sustained completed-game throughput and resource
telemetry rather than treating 32 as a floor. Bias policy-leaf inference toward
the higher-capacity Blackwell GPU while retaining the RTX 3080 Ti as bounded
spillover; the exact leaf split must be receipt-backed and must not overload
either device. Do not interrupt the current in-flight collection merely to
apply this correction: activate the selected topology at the next clean managed,
append-only, receipt-backed collection boundary. Inzi-only/no-remotes and
RTP/search/MCTS-off behavior remain unchanged. Revision 296's worker-times-four
accounting still governs any later explicitly authorized remote readmission.
Typed source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 300-TRAINING, the owner selects the next Inzi production
topology as exactly 16 simulator processes with four `LibcgMultiEnv`
environments per process, for 64 concurrent environments. This supersedes
revision 299's open process-count selection and revision 297's 32-process
requirement. Keep Blackwell as the primary policy-leaf inference device and
the RTX 3080 Ti as bounded spillover. Do not interrupt the current in-flight
self-play phase; activate 16×4 only after that phase seals at the next clean
managed append-only boundary. Inzi-only/no-remotes and RTP/search/MCTS-off
behavior remain unchanged. Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 301-TRAINING, supersede revision 300's 16×4 selection so the
next Inzi collector restores the historical target of 96 concurrent local
game environments under multi-environment packing. The prior 96 profile used
96 singleton simulator workers at one game each; the packed equivalent is
exactly 24 simulator processes with four `LibcgMultiEnv` environments each.
Activate 24×4 only after the current self-play phase seals at the clean managed
append-only boundary. Keep Blackwell primary, the RTX 3080 Ti as bounded
spillover, Inzi-only/no-remotes, and RTP/search/MCTS off. Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 302-TRAINING, the owner clarifies that revision 298's isolated
Alakazam simulator-rules and auxiliary-head experiment is a separate goal,
not a live branch of this production goal. Preserve revision 298 and its
attachment as immutable historical provenance. The then-current revision-4
experiment gateway was
`goals/alakazam-elmo-rule-derivative/GOAL.md`
(`sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8`);
its sole typed canonical source is
`goals/alakazam-elmo-rule-derivative/contract.json`
(`sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2`),
bound to owner
attachment
`/Users/tsinzitari/.codex/attachments/8100a736-c437-41f2-b559-97f78d2e2f78/pasted-text.txt`
(`sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f`).
At revision 302 this root goal and
`state/alakazam-new-list-direct-policy-r241.json` retained
only a non-authoritative handoff and historical reference for that experiment;
they must not supply competing live experiment semantics. The separate goal
authorizes isolated Elmo implementation and evaluation only. It grants no
Inzi file, service, runtime, worker, collection, checkpoint, selector,
package, submission, propagation, restart, or production-activation change.
Revisions 300--301's production-topology decisions are unchanged.

Under revision 303-TRAINING, after the dedicated derivative completes Elmo
implementation/schema freeze, exact 2026-07-13 through 2026-08-11 UTC
re-featurization and census, branch adjudication, and verified quarantined Inzi
shard staging, hand bootstrap/training to Inzi's `cuda:1` RTX PRO 5000
Blackwell at one clean receipt-backed boundary. Preserve current healthy r274
work until its collection/shard, optimizer/adapter, checkpoint, and commit are
durable; do not stop early or switch mid-unit. Then fully pause exactly
`pokebot-alakazam-r274-rl.service` and the r274-specific
`pokebot-alakazam-r274-rl-submission-boundaries.service` through user systemd,
leaving the shared `pokebot-kaggle-submission-queue.service` unchanged. Seal
the exact r274 parent,
optimizer, registry, service definitions, corpus, schema, frozen tensors,
Blackwell ABI/capacity, readiness, activation, and rollback receipts before
staged shards become training-eligible. The derivative becomes the sole active
Alakazam training/collection lineage; “overwrite” is a logical managed-lineage
handoff, never destructive mutation or deletion of r274 artifacts.

After successful Blackwell bootstrap and candidate validation, build exactly
one checksum-exact Kaggle package on the exact new Alakazam list, binding the
checkpoint, deck, runtime, simulator/catalog, feature/target/checklist schemas,
Matchup Adapter inventory, package parity/smoke, visible label, and single-use
authorization. Durably enqueue it `first_if_allowed` under the existing quota
and spacing guards. A pending quota/spacing entry is sufficient for the next
step; do not wait for upload or score, do not claim submission without an
upload receipt, and never lose, mutate, or falsify the checkpoint on delay or
failure.

Once that queue receipt is durable, start derivative self-play across every
known host proved available, eligible, and checksum-compatible by the same
frozen fleet inventory. Bind per-host package parity, explicit GPU/worker
routing and resource caps, managed service definitions, global collection and
game/lease identities, append-only shards, and duplicate suppression. Inzi
Blackwell remains the sole learner/optimizer. No old r274 training or
collection may run concurrently, and no unreceipted host drop, device fallback,
duplicate game, resource-cap weakening, serving-selector change, or shared
queue-service change is allowed. No runtime or service change is performed by
recording revision 303. The dedicated derivative semantics and closed receipt
inventories remain solely canonical in
`goals/alakazam-elmo-rule-derivative/contract.json`; the production handoff is
owned by `state/alakazam-new-list-direct-policy-r241.json`.
The current dedicated revision-8 gateway is
`goals/alakazam-elmo-rule-derivative/GOAL.md`
(`sha256:3e710a6f474e096e2562c8a42d6c886e78009baf622c9fd6cb68901657c7ced4`)
and its sole typed contract is
`goals/alakazam-elmo-rule-derivative/contract.json`
(`sha256:b522af1617f02a49522302947f1a4841ef24db7213f0e2ea8abeaba1332fb2cc`).

Under revision 304-TRAINING, the owner terminates this root task's r274
production cycle after the currently active iteration 1 update. Finish
iteration 1 without interrupting or recollecting its in-flight exact
collection: seal exactly 8,196 games with every existing seat, H10 Marnie,
frozen submission-55378392, direct-policy, RTP/search/MCTS-off, adapter,
combo-off, and receipt invariant; then run its ordinary retained-head
optimizer, one isolated Matchup Adapter continuation epoch, exact revision-284
two-opponent formal holdout, and durable `iter_00001` commit.

Build and upload exactly one new `first_if_allowed` direct-policy Kaggle
submission from that exact durable iteration-1 learner, with the exact new
Alakazam deck, runtime-on trained Matchup Adapter bank/tree, and RTP, search,
and MCTS absent. The upload must receive its own immutable request, package,
single-use authorization, queue/attempt, and accepted submission-ID receipt;
pending quota or spacing waits and may not be called submitted. After that
accepted upload receipt, stop the managed r274 loop cleanly. Do not collect
iteration 2 or execute former refresh/submission boundaries 5, 10, 15, or 20.
This supersedes revision 285's remaining r274 horizon while preserving its
already completed bootstrap submission and every immutable iteration-0 and
iteration-1 artifact.

Because the current interpreter was launched under the superseded horizon,
arm a checksum-bound managed boundary fence before iteration-1 collection
seals. The fence must prevent iteration-2 dispatch without racing a
post-commit poll and may restart only the managed r274 trainer at the sealed
iteration-1 collection boundary if required to load the fence. Never
terminate or interfere with an interactive session. Revision 304 governs only
this root r274 loop; revision 303's separately owned derivative goal remains
separate and receives no runtime authority from this instruction. Typed source
remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 305-TRAINING, every future pure-RL optimizer cycle repeats a
deterministic contiguous packing step after its source collection shards are
sealed and before baseline preparation or backpropagation.  This RL replay
pack is distinct from the revision-279 expert pack and from the legacy
eight-part Python-object replay cache.  It contains the exact accumulated
training games, decisions, targets, masks, variable-length offsets, source
identities, and checksum-bound Matchup Adapter ticket/route metadata needed by
the ordinary full-model optimizer and the isolated adapter continuation.

The step always runs locally on Inzi from immutable local shards, using
configurable deterministic multiprocessing at 1/2/4/8/16/32 workers; Elmo,
Bert, and LAN transfer have no role.  Worker count is not capped at eight: use
the fastest supported count proven by an isolated local Inzi benchmark for the
actual corpus and available RAM/I/O bandwidth, but never change canonical
ordering or tensors.
Merge in source order, cache the completed pack for optimizer epochs and later
refreshes, and require exact serial parity plus identical seeded one-step
optimizer results.  Keep batch size, shuffle, losses, weights, masks, and
optimizer semantics unchanged; larger batches require a separate owner
decision.  Preserve the serial loader as fail-safe.  Do not modify or restart
the currently active revision-304 optimizer; activation is allowed only for a
future run at a sealed receipt-backed boundary after validation.  Typed source
remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 306-TRAINING, select exactly 16 local Inzi preparation workers
for the revision-305 contiguous RL replay packing step on subsequent training
starts.  Retain 1/2/4/8/32 only as supported diagnostic configurations; do not
auto-select them for production.  This does not restart, reconfigure, or alter
the currently active revision-304 optimizer.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 307-TRAINING, finish the already active r274 iteration-1
ordinary optimizer and its required one-epoch Matchup Adapter continuation,
but skip both the revision-284 formal holdout and the research-control
measurement for this terminal iteration.  Do not interrupt the optimizer,
discard its candidate, recollect games, or claim any measured evaluation pass.
Commit the exact trained learner as durable `iter_00001`, build and upload the
single revision-304 `first_if_allowed` direct-policy Kaggle submission with
RTP, search, and MCTS off, then stop the managed r274 loop before iteration-2
collection.  Preserve all earlier diagnostic and measured evidence unchanged.
This is a terminal evaluation waiver for this exact root r274 iteration only;
it does not alter future evaluation defaults or the separately owned derivative
goal.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 309-KAGGLE, submit one additional checksum-exact copy of the
immutable revision-195 NO-RTP Alakazam package from Kaggle submission
`55378392`.  Reuse bundle
`sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145`
and checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`
without changing any archive member, deck, runtime, Matchup Adapter, search,
RTP, checkpoint, selector, or training service.  The copy remains
`first_if_allowed` and uses the exact visible label
`r195 iter 21 261d367e131e NO RTP`.  Queue it immediately under the existing
daily quota and four-hour spacing rules with a fresh one-shot authorization
and immutable upload receipt.  This copy does not replace, retry, relabel, or
mutate submission `55378392`.  This completed as Kaggle submission `55468965`
at `2026-08-12T23:38:18Z`; evaluation was pending at activation.  Typed source:
`state/alakazam-r195-no-rtp-additional-copy-r309.json`.

Under revision 298-TRAINING, stage a separately versioned simulator-rules
representation and auxiliary-head overhaul for Alakazam as an **isolated
Elmo-only experiment**.  Do not change, restart, activate, package, propagate,
or otherwise touch Inzi or production.  Preserve revision-195 NO-RTP as the
immutable weight parent and the current exact r241/r274 policy as the baseline.
Any implementation is a versioned derivative whose zero-gated/layer-off mode
reproduces the baseline logits exactly; it may not rewrite the parent
checkpoint, current baseline, selector, packages, or active head logic in
place.

The pinned competition simulator and its legal option list are ground truth;
official paper rules are secondary where they differ.  Do not invent attacks
the simulator omits when an effect cannot resolve, including full Bench,
deck-zero draw, or opponent-handCount-zero effects.  Mega Zygarde ex
Nullifying Zero resolves automatically left-to-right with no target-order
choice and simultaneous KOs.  Reproduce the simulator's sequential Prize and
promotion prompts after simultaneous KOs, calling a result a draw if both
players ultimately reach zero Prizes.  Mega Pokémon ex yield three Prizes,
ordinary Pokémon ex yield two, ordinary Pokémon generally yield one, subject
to explicit modifiers; `megaEx` takes precedence when both `ex` and `megaEx`
are true.  The "next Alakazam line" remains the replacement attacker on the
Bench only, never the current Active line.

All policy, value, Fusion, rule-adapter, checklist, target, and public-belief
search inputs use only the acting player's public information set.  Exact
opponent hand identities, opponent deck order, and unrevealed Prize identities
are never runtime inputs.  Privileged hidden fields may appear only in a
separately typed named-belief target payload and may not generate actions,
policy teachers, action filters, rollout branches, or counterfactual values;
one realized true-environment trajectory/reward is allowed.  A metamorphic
test must vary every privileged opponent-card identity while holding public
state/history and seeds fixed, then prove sanitized features, policy logits,
and public-belief decisions bit-identical.

Phase 1 is an Elmo collision census over the pinned simulator and exact
r274/new-list data.  Record the canonical public-observation hash, current
feature/token hash, complete semantic option key, and simulator
successor/event-chain hash for every legal option.  Fail on equal model
encodings with distinct simulator successors or outcome distributions; classify
intentional hidden-information equivalence, intentional permutation
equivalence, simulator-legality proxy, and genuine public-state/option
non-identifiability, including frequency and action-change risk.

Phase 2 adds a parallel zero-gated public-rule adapter, not an in-place parent
mutation.  It represents public HP/max HP, evolution stack/preEvolution and
appearThisTurn, typed-energy units and attached card identities, effective Bench
maximum, turn-action/resource flags, public terminal reason, and visible
looking/search menus.  It encodes select context/type, contextCard/effect
source, bounded exact NUMBER values with overflow failure, min/max and
remaining damage/energy budgets, Option.count, normalized SKILL
physical-source/serial binding, semantic source/target/area/slot/attachment,
and a stable simulator discriminator only after semantic fields still collide.
It must not blindly embed global serials or candidate ordinals, and it must
preserve option-permutation equivariance unless order is simulator-semantic.
Structured public card/attack metadata feeds a zero-initialized residual; text
hashes are not a rules parser, and pinned-simulator transitions supply targets.

Phase 3 derives exact public simulator targets for option semantics, typed-cost
attack readiness, immediate damage/counters/draw/discard/energy/Bench/KO,
post-modifier Prize yield, turn-resource use, deterministic prompt chains,
terminal reason (including simultaneous-KO draw), and deck-out/forced-draw
timing.  In the derivative, repair `lethal_threat`, `prize_race`,
`action_utility`, `game_phase`, `opponent_hand`, `opponent_remainder`, and
`terminal_conversion` against those targets exactly as bound in the typed
source.  This authorizes no production learned-head, Fusion, OwnDeck, or
Matchup Adapter mutation.

Phase 4 requires all eight checklist channels at every scored factorized stage
and in evaluate-only forced turns: value/evidence, availability mask, exact
public provenance, unavailable/unknown reason, signed normalized vector,
applied gate, and post-cap residual.  Unknown or hidden evidence is neutral
zero, never guessed.  `bench_prize_exposure` (Q5) and
`immediate_disruption_outcome` (Q6) remain trace-only at exact zero gate until
their predicates are separately calibrated and explicitly enabled; do not
claim full logit coverage for either.  Preserve Q3's Bench-only replacement
line contract.

Phase 5 requires engine-backed and metamorphic tests for bounded NUMBER,
distinct SKILL sources, semantic attack/prompt collisions, energy/damage
budgets, evolution/turn flags, typed special energy, Prize yields and
modifiers, deck/Bench/hand legality, Nullifying Zero ordering, simultaneous-KO
Prize/promotion/draw behavior, initiating-action prompt-chain credit, hidden
state invariance, legal masking, permutation/padding, finite residual caps,
and exact layer-off baseline logits.  The owner attachment is
`/Users/tsinzitari/.codex/attachments/7b1d4464-b2f6-4fcc-8a6e-52abc35e3aaf/pasted-text.txt`
(`sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb`).
Typed source remains `state/alakazam-new-list-direct-policy-r241.json`.

Under revision 251-TRAINING, correct r241's activation topology without
changing its deck, guide, simulator, ten-update cycle, peak-r195 head/adapter
semantics, exact public mix, PTCGReplay deferral, or one-terminal-submit
boundary. The typed r241 owner source is immutable intent only: it declares no
derived readiness or operation-authorization state. The only activation path
is the ordered DAG: immutable owner intent, checksum-bound source and baseline
payload snapshots, offline host receipts, one logical create-only activation
overlay, then the managed services. The logical overlay must bind both hosts'
source, baseline, official-libcg, peak-r195, and remote-worker evidence; its
Inzi and Elmo publications are byte-identical and have one shared SHA-256.
The static source-snapshot registry remains pending intent and cannot itself
authorize a service. The overlay's checksum-bound owner-start authorization is
the sole execution authorization proof.

Clarify the direct/no-MCTS boundary precisely: it applies to the r241 learner,
pinned H10 Marnie direct opponent, target generation, terminal package, and
submission. It does not change the established frozen non-H10 diverse public
opponent packages or selectors preserved by revision 245. Do not add a public
search firewall, relabel those packages, or alter concurrent MCTS work. The
new topology remains staged; no activation, service start, training, or
submission follows merely from recording it. Typed source:
`state/alakazam-new-list-direct-policy-r241.json`.

Under revision 242-Kaggle, supersede only revision 240's high-confidence
frozen-direct threshold for the pending revision-235 replacement package. A
complete, legal, precomputed frozen-r195 direct action qualifies when **every**
selected factorized-stage probability is finite and greater than or equal to
`0.80` (inclusive), rather than `0.90`. A missing, malformed, non-finite, or
below-`0.80` selected stage remains ambiguous and routes to the unchanged
two-lane bounded MCTS route. The revision-240 `0.90` threshold draft and every
preflight that validates that superseded threshold are historical and ineligible
for the current R235 gate or immutable binding; the high-confidence/adaptive
MCTS regression must be reissued against `0.80`.

All other revision-240 behavior remains unchanged: high-confidence direct is
still non-degraded, starts no child, and calls no MCTS/select/search/model/
simulator; an already-running child receives only the one history-only
`note_direct_action` synchronization IPC. Deterministic continuation remains
bounded to eight validated steps, ambiguous MCTS remains exactly two lanes with
the 2.0-second child / 4.0-second parent limits, adaptive ≥8/leader×3/both-lane
early stop and 32-backup hard stop, and zero-backup direct fallback. Revision
242 preserves r234 containment, r236 libcg, the 65,536 cap, r238 resources,
the exact one-shot R235 label/upload boundary, historical r228 evidence, and
the separate r229/BO1000 lifecycle. Typed source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

For ambiguous MCTS under the same revision, simulated rollout expansion stops
at a terminal, chance boundary, or actor change away from our root seat (our
end-turn/opponent transition). An actor-change leaf is value-evaluated but has
no expanded legal actions or children; MCTS never selects or plans an opponent
action. Receipts report `actor_change_boundary_leaf_count`,
`chance_boundary_leaf_count`, and `boundary_leaf_count`. Deterministic
continuation already stops at that same actor boundary.

Under revision 243-INSPECTOR, reconstruct Replay Model Inspector traces at the
selected physical-game boundary. Selecting one checksum-attested submission
and replay game starts or joins one Elmo materialization covering every
selectable own-agent decision step and factorized stage in that game. Each
changing causal decision is encoded at most once for that materialization;
factorized stages reuse its exact model state. Changing only the displayed
step or stage reads the materialized result or joins the same in-flight game
job and must never start an independent GPU reconstruction. A newly selected
physical game or submitted model receives its own materialization.

The browser fetches only the selected address; Elmo, not Bert or the browser,
owns the full-game job. Nonresident packages use one checksum-bound worker
that streams bounded raw trace frames back to Elmo independent of whether the
disposable cache can commit. Its heartbeat resets a 240-second idle watchdog,
but there is no total elapsed worker deadline and a worker failure may not fan
out into per-address workers.

Physical-game identity is not an episode number alone. Reuse requires the same
submission, archived own seat, replay digest, submitted bundle, runtime source
tree, checkpoint, matchup tree, parity receipt, baseline request semantics,
and inspector trace schema. Elmo may store only the resulting derived JSON in
a bounded private `/tmp` cache. Entries must be checksum-verified,
identity-bound, symlink-safe, and atomically published; a mismatch, corruption,
partial write, failed trace, or changed source digest is a miss and can never
authorize runtime/model substitution. Playground head-scale counterfactuals
remain separate computations and may not poison the baseline game cache.

Preserve Elmo `cuda:0`, one serialized GPU materialization, one resident model
maximum, exact package-local guide and adapter behavior, request-local runtime
state restoration, r237's lack of a browser deadline,
`recomputed_not_historical`, loopback/read-only service boundaries, and zero
training, selector, or submission authority. Bert remains transport-only. Its
direct-LAN trace gateway has a bounded five-second tunnel connect, 30-second
post-header body-idle wait, 8 MiB response cap, and at most four long trace
waiters, but no response-header wall-clock deadline while Elmo reconstructs.
This supersedes r222/r237's no-whole-game-prefetch behavior only for a game the
owner explicitly selects; unsolicited cross-game prefetch remains forbidden.
Typed source:
`state/replay-model-inspector-physical-game-materialization-r243.json`.

Under revision 244-LIBCG, correct only the official `libcg` SearchId identity
interpretation for the new r235 Kaggle replacement and the next r239 BO1000
package. A numeric `SearchId` namespace is scoped to its distinct
`AgentStart()` handle: two isolated handles may therefore both report first raw
SearchId `0`. Global raw-SearchId integer uniqueness is not an isolation proof
and is not required. Each two-lane receipt instead requires exactly two arenas
and `SearchBegin` calls, two distinct per-lane handle identities, both
per-lane SearchId chains, and two distinct composite
`(handle_identity, first_search_id)` states. A missing handle/chain or a
duplicate composite state fails closed; a repeated raw SearchId on distinct
handles does not. The public ordered composite-state receipt entries are
exactly `{"lane_id": <0 or 1>, "handle_identity": <opaque AgentStart identity>,
"first_search_id": <handle-scoped native SearchId>}`; lane is reporting
context, while state uniqueness is the `(handle_identity, first_search_id)`
pair.

This correction preserves the r242 Kaggle high-confidence/direct-fallback,
continuation, bounded-search, containment, resource, 65,536-cap, canonical
r236-libcg, R235 single-upload, and historical-evidence rules. It also
preserves r239's exactly-two-lane topology, r233 outer game watchdog, and
pre-r234 BO lifecycle boundary. It authorizes no code execution, job, upload,
commit, managed-service action, training, selector, or BO1000 game. Typed
sources: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`
and `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 246-Kaggle, supersede only the ambiguous-decision root-action
selection rule for the pending r235 replacement package.  The revision-242
inclusive `>=0.80` high-confidence frozen-direct route remains first and
unchanged: it starts no child, so it cannot use this new exception.  Once an
otherwise ambiguous decision has entered its exact two-lane MCTS child, a
single returned proof of a **deterministic terminal win this turn** has
absolute root-action selection and early-stop authority over priors, visit
counts, and nonterminal alternatives.  It is a proof returned by the stock
simulator search, not a model-value prediction, policy confidence, heuristic,
or an exhaustive legal-action scan.  The proof must be backed into the shared
root tree and identify the exact current root observation fingerprint, complete
ordered legal fingerprint, root actor, legal root action, selected action, and
terminal winner.  The terminal result must be `win` for that root actor.

Every simulated action on the proof path must remain that root actor's action;
it may not cross an actor/opponent boundary, a chance boundary, unresolved
randomness, or any outcome that can throw, draw, or lose.  A chance/coin line,
an opponent-response line, a model-predicted win, a stale or malformed proof,
or a loss/draw/nonterminal terminal result has zero override authority.  The
parent validates the proof against its current exact root before acting; a
claimed proof with a stale/malformed binding follows the existing child
protocol-fault containment rather than selecting its action.  Two lanes still
must be initialized and cleaned under r238/r244, but a second independent
winning proof, both-lane progress, the normal `>=8` backup / leader-times-three
gate, or an exhaustive scan is not required once one exact deterministic
terminal proof is returned.  All exact child resources and reservations still
must be cleaned before the parent returns the selected legal root action.

This adds the receipt-backed stop reason
`proven_deterministic_terminal_win_this_turn` and a focused regression/binding
gate.  It preserves r234 containment, r236 libcg, r238 resources, r242
scheduler/direct fast path, r244 handle-local identities, the 65,536 action
cap, the exact one-upload R235 boundary, historical failed-run evidence, and
every r229/BO1000 lifecycle boundary.  Typed source:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 249-BO1000, replace only r229's unsafe in-process native-search
lane boundary.  The frozen r195 model and the one logical shared MCTS tree stay
in the authoritative game process.  Exactly two persistent simulator worker
**processes** each own one official revision-236 `libcg` `AgentStart` handle;
no native `SearchBegin`, `SearchStep`, `SearchRelease`, or `SearchEnd` call runs
in a Python thread that the game process cannot reap.  Each request and cleanup
has a finite parent deadline.  A hung, crashed, malformed, or cleanup-failed
lane causes only its exact owned child to be reaped; interactive sessions and
managed services are never signal targets.

Because a replacement native handle may represent a different hidden or random
world, no partial tree from a failed attempt is reused.  Reopen fresh two-lane
workers and retry the complete decision once from the same exact root
observation, complete ordered legal space, frozen model state, and historical
eight-second r228 search budget.  A successful retry is ordinary fully backed
two-lane MCTS authority.  Only after that bounded retry also fails may the game
continue with the already computed, legal, same-state frozen-r195 direct
counterfactual.  Such an exhausted-recovery fallback is explicitly degraded,
has no MCTS-change or meaningful-change credit, and records both attempt faults,
process identities, reaping outcomes, latency, and exact fallback action.  It
must never be relabelled as searched MCTS.  The clean full-game launch preflight
requires zero exhausted-recovery fallbacks; the final review separately reports
all lane restarts, recovered searches, exhausted fallbacks, and their effect on
throughput and evidential coverage.

Revision 249 preserves r230's complete 65,536-action enumeration, r236's four
official platform libraries, r239's exact two-lane topology, r244's
handle-scoped SearchId composites, and r233's outer per-game
watchdog/requeue/quarantine.  It does not import any Kaggle broker, queue
cleanup, high-confidence bypass, adaptive stop, continuation plan, terminal-win
override, or r240/r242/r246 search-policy change.  Evaluation games remain
training-ineligible and have no selector, serving, promotion, checkpoint,
managed-service, RTP, or Kaggle authority.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 250-BO1000, supersede r239/r249's active exactly-two-lane
topology for the next r229 package and evaluation.  Use serial MCTS for now:
the authoritative game process retains the one frozen r195 model and one
logical MCTS tree, while exactly one persistent owned child process holds one
official-r236 native simulator handle.  No two `libcg` search calls may be in
flight concurrently, and the serial decision receipt requires one arena, one
`SearchBegin`, one handle-scoped SearchId chain, and one
`(handle_identity, first_search_id)` state.  Preserve the historical
eight-second r228 search budget, r230's complete 65,536-action enumeration,
r233's outer per-game watchdog/requeue/quarantine, r236's official native
library set, and r244's handle-scoped SearchId interpretation as applied to
that one handle.

Every native request and cleanup remains parent-bounded.  If the child hangs,
crashes, returns malformed output, or fails cleanup, reap only that exact
owned child, discard the complete partial attempt tree, open one fresh serial
child, and retry the same complete root once.  A successful retry is ordinary
serial-MCTS authority.  Only retry exhaustion may use the precomputed legal
same-state frozen-r195 direct counterfactual; it remains explicitly degraded,
gets zero MCTS-change or meaningful-change credit, and must carry complete
fault/reap telemetry.  A clean full-game launch preflight still requires zero
exhausted-recovery fallbacks.  The r239 two-lane and r249 two-process packages,
preflights, and partial results are historical and execution-ineligible for
this serial run.

The bounded serial contract also applies after the worker response reaches the
coordinator.  A consumed worker-error row must clear that lane's in-flight
reservation before the post-deadline drain, and a cleanup-error row is the
terminal cleanup response for that lane.  The coordinator may preserve either
error and retry the complete root, but it may never wait for a second response
to the already-completed command.  Likewise, once a successful native step row
has arrived, its in-flight reservation must be cleared before parent-side
packet construction or model evaluation, because an exception there cannot
produce another native response.  This is narrow BO-only completion accounting
required to make r249/r250 recovery reachable; it does not import the r234
Kaggle broker, direct fallback, or queue lifecycle.

Only after a checksum-bound serial package completes the clean full-game gate
may r229 begin a separately versioned investigation of process-parallel node
evaluation during tree construction.  That follow-on must keep authoritative
tree coordination in the parent, give each native worker process its own
handle, and forbid concurrent native calls within any one worker.  It has no
current package, preflight, game, fleet, or action authority and may not change
the Kaggle-specific r228 broker, queue, runtime, or submission lifecycle.
Revision 250 otherwise preserves r249's bounded process ownership/recovery and
excludes all Kaggle high-confidence bypass, adaptive-stop, continuation,
terminal-win, broker, and cleanup policies.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 252-MCTS, add the same simulated-leaf boundary rule to the
pending r235 Kaggle replacement and to the separately owned r229 BO1000
experiment.  This rule does not raise the MCTS root ceiling: an exact real-root
ordered-action count at or below 65,536 remains completely enumerated with no
threshold pruning.  For the Kaggle r235 replacement, a real root above 65,536
still hard-fails.  For r229 BO1000 only, an exactly counted over-cap real root
must not enter MCTS; it uses the frozen-r195 factorized direct policy's legal
same-state action and records `oversized_direct_fallback`.  That BO action gets
zero search, MCTS-change, or meaningful-change credit, is not a retry or
retry-exhaustion fallback, and may not silently contribute to the reported
MCTS rate.  This is the explicit fail-closed accounting required by r230 while
allowing a normal physical game to continue through pathological combinatoric
prompts.

Within private simulation, every recognized chance context is a hard
pre-random value boundary.  In addition, any stochastic resolution with more
than ten possible outcomes, or whose equiprobable outcome probability is at
most `0.10`, is a hard pre-random value boundary even if it was not previously
classified by a named chance context.  The search must not sample, choose,
plan through, or assign action authority at such a boundary.  A deterministic
internal simulated choice with more than 64 complete ordered outcomes is also
a value-only boundary.  Any inert singleton used solely to batch one frozen
value evaluation is not a legal-action candidate, child, rollout step, or
continuation-plan action.  Receipts must distinguish recognized-chance,
stochastic-cardinality/probability, and deterministic-internal-cardinality
cutoffs and prove zero boundary expansion.

For Kaggle this preserves r242's inclusive `>=0.80` direct-before-child path,
r246's deterministic same-root-actor terminal-win exception, the exact
two-lane topology, adaptive limits, contained fallback, r236 native identity,
and R235 one-upload boundary.  An r246 terminal proof cannot cross any new or
existing stochastic boundary.  The already sealed revision-246 archive is
historical preflight evidence and is upload-ineligible; a fresh checksum-bound
r252 stage, failed-replay regression, fault gates, physical full-game/resource
preflight, and immutable binding are required.  For BO1000 the same leaf
cutoffs are owned by the r229 typed source and preserve its current serial
process-owned native lane, retry/reap policy, r233 outer watchdog, and explicit
Kaggle-lifecycle exclusion.  Typed sources:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json` and
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 253-BO1000, supersede r250's invalid continuous-trajectory
serial-search mechanics while retaining one authoritative parent tree, one
owned child process, and one official-r236 `AgentStart` handle with no
concurrent native calls.  A serial MCTS decision consists of repeated
independent root rollouts.  For every rollout, call `SearchBegin` from the same
exact current physical root, use the parent tree to select and traverse legal
actions, stop after one leaf is expanded and frozen-value evaluated or an r252
value-only boundary is reached, back that value through the selected path,
then boundedly release/end that rollout.  Reopen the exact root for the next
rollout and continue until the decision deadline or explicit rollout ceiling.
One continuous native `SearchStep` trajectory is never a substitute for
multiple root rollouts.

The parent retains the sole logical tree across successful rollouts within one
attempt; native SearchIds and simulator state never carry tree or action
authority between rollouts.  A failed, timed-out, malformed, or cleanup-failed
rollout fails the complete attempt: reap only the exact owned child, discard
the entire partial tree, reopen one fresh process/handle, and retry the exact
complete root once under r249/r250.  No successful rollout prefix is reused
after such a fault.  Retry exhaustion alone may use the existing degraded
same-state direct fallback.  r252's chance/large-internal value boundaries and
BO-only over-cap-root direct boundary remain unchanged.

The stopped first r252 BO launch's 13 completed games and 741 searches are
immutable invalid-diagnostic evidence only.  Because the one-lane continuous
trajectory could visit only one root edge, its observed zero action changes
were structurally forced and carry no BO result, MCTS-rate, search-effect, or
model-comparison authority.  Before a new BO1000 launch, focused regression
and a clean full-game receipt must prove multiple independent root
`SearchBegin` rollouts, parent-tree visit accumulation across rollouts,
selection of more than one root edge in a constructed multi-action case,
bounded per-rollout release/end, zero leaked reservations, and a constructed
case where backed values can change the selected action.  Preserve r233's
outer watchdog, r236 library identity, host-local package/model/data plane,
training-ineligibility, and every Kaggle lifecycle boundary.  No Kaggle r228
source, package, preflight, or submission authority changes.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 254-BO1000, the owner fully abandons the running r253
full-MCTS BO1000 after its clean managed stop at 45 completed games.  Preserve
all completed games, attempts, logs, checkpoints, package identities, and the
owner-abandonment receipt as immutable, training-ineligible diagnostic
evidence.  The incomplete 45-game prefix may support the explicit owner stop
and search-quality diagnosis, but it is not a 1,000-game efficacy result and
must never be relabelled as one.  Do not resume, refill, or replace the 955
unplayed games, and do not start the gated process-parallel full-MCTS follow-on.

The next search direction is separately versioned conservative selective
tactical proof search, not a smaller copy of r253 MCTS.  Frozen r195 direct
policy remains the default action.  Search may be invoked only at a genuinely
conflicted branching decision under a checksum-bound trigger using the direct
complete-action distribution (including top probability and top-one/top-two
margin).  Initially it may override direct policy only with an exact
stock-simulator proof of a deterministic terminal win this turn for the root
actor.  Learned leaf value, visits, priors, partial trees, and unproven
nonterminal lines have no override authority.  Chance, unresolved randomness,
opponent/actor transition, and deterministic internal fanout above 64 remain
hard boundaries.  Every simulated action must be represented in the frozen
model's temporal previous-action history; the r253 omission of simulated
action tokens is preserved as diagnosed historical behavior, not copied.

Only implementation, focused tests, exact-package preflight, and a small
training-ineligible shadow/pilot are authorized by this record.  Trigger
thresholds, time budget, pilot size, and any broader action authority require a
fresh typed configuration and receipt before execution.  No BO1000, Kaggle,
training, serving, selector, promotion, checkpoint, or model-update authority
is granted.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 256-BO, the owner abandons all search work for now, including
the proposed revision-255 conflict-triggered tactical proof search.  Do not
implement, package, preflight, pilot, launch, resume, or refill MCTS, tactical
proof search, or another search variant.  The uncommitted r255 prototype is
removed without execution and has no result, action, package, or promotion
authority.  Frozen r195 direct policy is the only action authority in this
evaluation line.  Preserve the stopped r253 45-game prefix and abandonment
receipts as immutable diagnostic evidence.  This changes no Kaggle, training,
serving, selector, promotion, checkpoint, or model state.  Any future search
work requires a new explicit owner revision.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 257-BO, the owner separately reopens implementation and local
testing of a **shadow-only goal-directed tactical sequence planner**.  This is
not a revival of MCTS and does not authorize a package, preflight, pilot,
BO1000, fleet game, Kaggle action, serving path, selector change, promotion,
checkpoint/model update, or training data.  Frozen r195 remains the only real
action authority.  The side planner may compare a proposed first action with
the exact same-state r195 direct action and emit an auditable certificate, but
it may never dispatch or override either an evaluation or live-game action
under this revision.

The planner has two distinct research goals.  An exact terminal-win goal
searches only deterministic same-root-actor transitions within the current
turn and may label success only when the simulator reports a terminal win for
that actor.  SME tutor/resource goals are public-fact shadow goals only.  A
tutor/search-card transition is an information/re-observation boundary: the
planner may not inspect or predict the hidden deck/prize realization beyond
it, and may continue only after a real recorded observation explicitly exposes
the selectable deck cards.  The Alakazam SME guide and frozen policy may
define goals and order candidates; the learned tactical-outcome head may be
logged as a hint, but no learned head, value, prior, visit count, partial path,
or SME score is proof.

Use policy-ordered limited-discrepancy search with small typed depth, node,
discrepancy, wall-time, and complete-action caps.  Stop before every chance or
stochastic transition, hidden-information boundary, actor/opponent or turn
transition, and deterministic complete ordered action space above 64.  Every
simulated step must bind the source observation/legal-order fingerprints and
must install the actual simulated action token into the next model-history
state.  Native simulation is allowed only through an owned bounded child
process; an in-process native backend is rejected.  Deterministic in-process
fixtures are allowed only through an explicit test-only marker.  Any fault,
deadline, malformed candidate, illegal action, fingerprint/history mismatch,
or boundary yields no proposal authority and leaves r195 unchanged.

All work stays in new sidecar/test paths plus this typed contract.  Do not
touch, stop, restart, reconfigure, resource-reduce, or otherwise interrupt the
healthy Alakazam refresh or any of its managed services.  Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 258-TRAINING, the owner authorizes a separately versioned,
direct-policy Alakazam successor that makes an exact-starting-list
`OwnDeckLedger` a match-scoped causal side store.  Build and validate the
isolated successor now, but do not alter the active r241 learner, checkpoint,
feature/corpus identity, selector, package, submission, or managed services.
The successor may enter implementation/training activation only after the
Alakazam refresh has a receipt-proven terminal completion; never stop, restart,
or migrate the healthy refresh merely to prepare this successor.

Update the ledger once per real acting-seat observation and freeze that
snapshot across every factorized stage of the observation.  It starts from the
known exact 60-card multiset and reconciles current own hand, board,
attachments/evolution stack, discard, known revealed prizes, recovery, and
other causally visible own-zone movements.  Face-down prizes and unrevealed
deck order remain unknown.  Encode per-card conservative draw-pile lower and
upper bounds, expected availability only when its assumptions are explicit,
exactness/integrity masks, deck and unknown-prize counts, and a prompt-local
candidate multiset when an actual `select.deck` or looking observation exposes
it.  Gameplay deck-search/tutor actions are ordinary learnable observations;
the no-search rule forbids tree search/rollouts and hidden realization, not
learning from real deck-search decisions.

Inject one canonical ledger representation into the shared model state before
policy, value, auxiliary/strategic heads, option decoding, and Fusion branch.
Thus every existing head can consume it without duplicating subtraction logic;
option candidates additionally receive typed ledger lookup features at visible
tutor stages.  Preserve an exact disabled/absent-input parent bypass and keep
all new weights, tutor supervision, terminal-conversion supervision, and
Fusion routes dormant until their own causal-coverage, gradient, calibration,
local/remote/replay parity, and bounded-influence receipts pass.  Immediate
terminal conversion must be a selected-complete-action observed target, not a
renaming or upweighting of the current eight-frame prize `lethal_threat` label.

After receipt-proven Alakazam refresh completion, the r258 owner decision
authorizes the staged isolated migration, training canary, source-disjoint
evaluation, and receipt-backed activation sequence without another owner
decision.  Each stage still fails closed independently; production selector,
package, promotion, or submission authority exists only after the complete
post-refresh evaluation and promotion receipts commit.  Evaluation/Kaggle
games remain training-ineligible, and r241 remains immutable.  Typed source:
`state/alakazam-own-deck-ledger-successor-r258.json`.

Revision 259-TRAINING clarifies that Elmo must begin materializing the r258
successor inputs now as a separate expert-corpus side store for the next
training lineage.  Read only the checksum-pinned exact-20-day r241 source
manifest at
`/mnt/Main/main/poke-bot-agent/archive/expert-r241-20260722-20260810/current.json`
(`sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16`;
2026-07-22 through 2026-08-10; 91,253 validated episodes), select only acting
seats whose exact starting list is classified as Alakazam, and write causal
per-observation ledger, visible-tutor, and terminal-conversion records under
`/mnt/Main/main/poke-bot-agent/archive/expert-r258-own-deck-ledger-sidecar/2026-07-22_2026-08-10`.
Keys bind episode, seat, environment step, and observation fingerprint; daily
outputs are atomic, idempotent, compressed, checksum-receipted sidecars.  The
source corpus is read-only and may never be rewritten or relabelled.

The new bounded
`pokebot-own-deck-rollout-store-r259.service` may be installed and started on
Elmo now from the isolated source snapshot
`/home/admin/pokebot-own-deck-r259-src-de844af19ca6`; this grants no authority to stop,
restart, reconfigure, or reduce any existing managed service.  The generated
store is ineligible for the active r241 refresh and for evaluation/Kaggle
training.  It becomes eligible only for the post-refresh successor after
source-join, causal-local/remote parity, schema, count, and checksum receipts
all pass.  All other r258 activation and promotion gates remain unchanged.
Typed source: `state/alakazam-own-deck-ledger-successor-r258.json`.

Revision-259 operational reconciliation: the first immutable Elmo source
snapshot and lock passed their source audit, then the contained archive-native
smoke failed closed before output or service installation when the generic
replay converter surfaced an explicitly `INACTIVE` stale action echo.  Preserve
that source tree (`sha256:e879340e0e94c956e7783c88fc5576e7906821a66670ba17dd44bf530279dbf9`),
lock, and failure as immutable audit evidence; never replace them in place.
The replay-corrected source uses the `cc72a4e27f94` code identity and filters
only same-seat/same-step raw rows proven `INACTIVE` or `DONE`.  Every retained
decision must still be the exact indexed raw `ACTIVE` row with matching actor,
public-observation fingerprint, and action; malformed, missing, or mismatched
rows fail closed.  Its second immutable seal then failed the contained nonroot
smoke before output because 0600 origin files became root-only 0400 files.
Preserve that `cc72a4e27f94` seal and lock too.  The third immutable attempt
used the literal `cc72a4e27f94-a3` namespace and correctly published traversable
0555 directories, readable 0444 regular files, and only the launcher as 0555.
Its contained smoke passed and the new service began, but the first day failed
closed before a daily commit when exact member `87394115.json` reported
`TIMEOUT,DONE` with rewards `[None,1]` and the legacy converter tried to derive
a numeric winner.  Preserve its source/lock, zero-restart service evidence, and
uncommitted hidden partial shard.  The active retry uses the literal
`3c8a28a0afd9` namespace above.  An outcome is verified only from top-level
`DONE,DONE` plus finite zero-sum rewards; every other episode uses a separate
winner-free r259 projection that retains exact raw `ACTIVE` ledger/tutor rows
and masks every terminal, prize-closeout, knockout, and tutor-terminal target.
The fourth immutable source (`3c8a28a0afd9`) added that exact-member smoke, but
it built the member's full roughly 1,850-decision record before applying its
post-conversion sample bound and was correctly killed by the required 1 GiB
smoke cap.  Preserve that seal and failure too; it never restarted the service
or touched a committed output.  The active retry uses the literal
`b96413d63f68` namespace above.  Its exact-member path enforced the small
decision cap *inside* fallback conversion, but the seal still combined the
normal and exact-member checks in one 1 GiB smoke process.  It was superseded
before any contained smoke evidence, output publication, unit replacement, or
service restart; preserve its source tree and lock as immutable attempt-5
evidence.  The active retry uses literal namespace `de844af19ca6`.  It runs a
normal-record-only 1 GiB contained smoke and a separate production-sized 2 GiB
contained smoke that directly exercises `87394115.json`, uses only the
winner-free fallback, and proves every outcome mask.  Both preflights must pass
before the failed r259 unit may be replaced and started again.

Under revision 260-TRAINING, the owner gives highest priority to getting the
still-unstarted revision-241 Alakazam run working and explicitly directs that
last night's revision-258 `OwnDeckLedger` head structure be folded into its
next receipt-backed successor before the first training update.  This
supersedes revision 258's wait-for-r241-completion and no-r241-mutation rules
only for this pre-start successor boundary: no r241 worker was armed, no
training collection or update began, and the failed launcher check created no
runtime authority.  Preserve the complete `1c34`/H10-v8/peak-v6/r13/quartet/
overlay line and its installed-but-inactive managed files as immutable
historical evidence; never patch or relabel those artifacts in place.

The combined successor keeps the immutable revision-195 checkpoint as parent
and preserves all 19 existing architecture heads, all 18 active non-combo
Fusion-v3 routes and their fixed denominator, the physically present but
loss-off/route-off `combo_state`, the Matchup Adapter bank and `no_slot_change`,
the exact r241 deck and guide, official revision-236 simulator, exact
1,024+7,172 collection, ten-update horizon, two five-epoch refreshes, and one
terminal direct-policy submission.  Add the revision-258 shared public-causal
`OwnDeckLedger` adapter v2 at width 128 before policy, value, every existing
learned head, option decoding, and Fusion; add the eight-feature visible-option
adapter; and add the typed seven-output `visible_tutor_completion` and
six-output `terminal_conversion` heads with their own zero-safe bounded option
routes outside the inherited 18-route Fusion denominator.  Their masked
factual losses split one total auxiliary budget of `0.05` equally.  MCTS, RTP,
tree search, hidden prize/deck realization, guide runtime authority, fabricated
counterfactual labels, and evaluation/Kaggle replay training remain forbidden.

Use the checksum-bound Elmo revision-259 expert side store as the successor's
training source as soon as its exact 20 daily shards plus terminal source-join,
schema/count/digest, causal local/remote parity, and completion receipts pass.
The protected exact-20-day source remains read-only, and a partial or
unreceipted side store is never eligible.  Zero-safe migration must preserve
every inherited tensor and exact parent behavior before the new paths train.
A bounded pre-start training canary may train the shared adapter and two new
heads/routes from the sealed expert corpus; runtime influence requires its own
finite-gradient, coverage, calibration, bounded-influence, local/remote/replay
parity, and source-disjoint evaluation receipt.  Once those receipts, the
corrected typed parent-`FileIdentity` launcher check, a new immutable source,
host-bound H10/peak/image/preflight line, and a fresh byte-identical activation
overlay pass, activate the combined successor immediately without another
owner decision.  Typed source remains
`state/alakazam-new-list-direct-policy-r241.json`; it is the single owner of
this combined r260 activation contract, while the r258 manifest remains
preserved implementation and corpus provenance.

Under revision 261-TRAINING, clarify that Inzi is the sole managed training
host for the combined successor.  Elmo may read the protected expert corpus,
materialize the public-causal daily side store, build the deterministic joined
dataset, and run bounded no-network parity or disposable-executor checks, but
it may not train the learner.  After all 20 daily shards and terminal receipts
pass, copy the exact daily layout and joined dataset create-only to
`/home/inzi/poke-bot-agent/outputs/pure_rl/alakazam_new_list_direct_policy_r241/runtime/r260-own-deck-training-dataset`,
publish a typed transport receipt, and rehash every FileIdentity on Inzi.  The
managed Inzi trainer must consume only that local Inzi root through the
disk-backed exact-four-key streaming index; any `/mnt/Main/` or other Elmo
side-store path in its environment, command, plan, or receipt fails closed.
This clarification changes placement only and preserves every revision-260
architecture, evidence, schedule, direct-policy, and activation requirement.

Under revision 262-TRAINING, overlap the remaining Elmo materialization with a
safe prefix transfer to Inzi.  Copy only already committed immutable daily
directories—never a dot-prefixed temporary day—to the non-eligible staging
root
`/home/inzi/poke-bot-agent/outputs/pure_rl/alakazam_new_list_direct_policy_r241/runtime/r260-own-deck-training-dataset-staging-09848f04`.
Each daily transfer is create-only, rehashes `meta.json` and
`own_deck_rollouts.jsonl.gz` against Elmo, and seals that day read-only on
Inzi; append later days only after their Elmo atomic commit.  A partial staging
root grants no dataset, canary, training, runtime, or activation authority.
After all 20 days pass, the deterministic join and terminal receipts may be
created and the fully verified tree atomically promoted to revision 261's
canonical Inzi training root.  Never stop, restart, or reconfigure the healthy
r259 managed producer to perform this overlap.

Under revision 263-TRAINING, fold the revision-257 shadow tactical sequence
planner into this same still-unstarted Alakazam successor and train its
public-state sequence/outcome hint live throughout all ten updates.  This is
training-only shadow cotrain: direct policy remains the sole dispatched action
authority, and tactical traces, learned hints, SME goals, discrepancy paths,
or certificates may never override, label, or become a submitted action.
Preserve the revision-257 chance, hidden-information/tutor re-observation,
actor/turn, owned-child, 64-complete-action, history, and fingerprint
boundaries.  The revision-257 prohibition on model/training changes is
superseded only for this r241 successor's receipt-backed shadow cotrain; its
MCTS/RTP/Kaggle/selector/serving prohibitions remain in force.

Train the complete OwnDeckLedger sidecar surface live in every update as well:
the shared ledger and option adapter, visible tutor-completion head, and
terminal-conversion head retain their revision-260 causal masks and 0.05 total
auxiliary budget.  Add one zero-safe four-output tactical-sequence outcome
hint head with a separate 0.025 masked loss budget.  Collect at least 1,024
bounded public-state tactical shadow roots from the ordinary direct-policy
training games in each update, bind their source observation/legal-order and
simulated-history fingerprints, and train only on exact simulator terminal
facts or explicitly observed public SME-goal/boundary labels.  Fabricated,
hidden, evaluation, Kaggle, partial-path, value/prior, or unreceipted labels
fail closed.  Both five-epoch expert refreshes continue training every
applicable deck-sidecar target; tactical supervision is used there only when
a checksum-bound compatible trace exists.  Before activation require
zero-safe migration, finite-gradient/coverage/calibration, bounded-influence,
source-disjoint evaluation, live-cotrain accounting, direct-action parity, and
shadow planner latency/reliability receipts.  No additional owner decision is
required after those gates pass.

Under revision 264-TRAINING, use the immutable revision-195 ladder-proven
Alakazam model checkpoint (`sha256:261d367e…9cc3a`) as the weight parent and
zero-safely adapt it to the r260/r263 deck-ledger and shadow-tactical system.
The 600.0 revision-195 RTP submission proves the shared model lineage, but its
RTP sidecar has no action, target, package, or serving authority in this
direct-policy successor.

Replace revision 241's ten-update/one-terminal-submit horizon with the owner's
established full cycle.  First run an exact 25-epoch expert bootstrap over the
checksum-pinned 20-day corpus and complete one first-if-allowed direct-policy
submission from that durable bootstrap checkpoint.  Then run exactly 25 RL
updates (`iter_00000` through `iter_00024`) with the unchanged exact 1,024
self-play plus 7,172 public-mix games, at least 1,024 direct H10 Marnie games,
and exact 4,098/4,098 seat split per update.  After completed updates 5, 10,
15, 20, and 25, run the small exact five-epoch expert soft refresh and submit
once from each durable refreshed checkpoint before continuing.  This yields
exactly six receipt-backed submissions total: bootstrap, then boundaries 5,
10, 15, 20, and 25.  No early gate exit, extra collection wave, milestone
skip, duplicate/retry copy, or unreceipted upload is allowed.  The r263 deck
sidecar and shadow tactical objectives cotrain through all 25 updates and every
applicable expert pass.  Update 25 is the terminal boundary; no `iter_00025`
collection is authorized.

Under revision 265-TRAINING, activate and train every architecture-present
ordinary model head and Fusion route in this successor.  In particular,
`combo_state` changes from present/loss-off/route-off history to exact masked
loss weight 0.025 and an enabled Fusion route, using only causal checksum-bound
labels.  Preserve the 18 already-active inherited routes and activate combo as
the nineteenth inherited route; keep the two r260 typed option heads/routes
active as already required.  The r263 tactical outcome head is fully trainable
and live as a shadow-planner ordering hint, but remains outside action Fusion:
that single deliberate boundary is required so the shadow planner cannot
become dispatched direct-policy authority.  Require per-head nonzero support,
finite gradient, optimizer-state, checkpoint inventory, causal-mask, runtime
influence, and terminal activation receipts; a silently dead or merely
architecture-present head fails closed.

Under revision 266-TRAINING, make the learned tactical-sequence outcome head's
Fusion route active in the direct-policy learner as well.  The bounded search
planner remains shadow-only and cannot directly dispatch, override, or supply
proof authority; its checksum-bound public-state traces supervise the learned
head, and that learned representation may influence ordinary direct-policy
logits only through the same typed, trained Fusion machinery as every other
active head.  This makes all 22 architecture heads and all 22 corresponding
routes active.  Log route-on versus route-off paired inference at bootstrap
and every five-update boundary, including action-change rate, top-action
margin delta, KL divergence, value delta, terminal-win/SME label calibration,
support, route magnitude, and latency.  Promotion requires finite, nonzero,
bounded influence with source-disjoint outcome evidence and no hidden-state,
proof, or direct-search authority.  Publish these impact receipts to the
training dashboard so the owner can inspect the tactical route separately and
alongside the full all-head model.

Under revision 267-TRAINING, add the exact frozen revision-195 600.0 ladder
submission package (submission `55378477`, bundle
`sha256:2f982f25…2000f9`, RTP sidecar `sha256:dde7b813…3aee`) to the
r241 public specialist research roster.  Run at least 128 games against that
exact package in every one of the 25 updates, with at least 64 learner-first
and 64 learner-second games.  These games are a named research cell inside,
not in addition to, the unchanged 7,172 public-mix total and do not reduce the
at-least-1,024 direct H10 Marnie cell.  The frozen r195 opponent may use its
checksum-bound RTP sidecar only as its own opponent action policy.  No r195
weight, RTP state, action, trace, hidden state, or package member may be loaded
into the learner, used as supervised target authority, or shipped in an r241
submission.  Require per-update package identity, seat, game-count, outcome,
and disjoint research-cell receipts and expose its results separately on the
dashboard.

Under revision 268-TRAINING, correct revision 267's research-opponent identity
to the owner's exact linked Kaggle submission `55378392`: the frozen
revision-195 **NO RTP** direct-policy bundle
`sha256:dfa8bfcc…b7145`, public score 500.4.  Submission `55378477`, bundle
`2f982f25…`, and RTP sidecar `dde7b813…` are not this research cell and have
no training-opponent, learner, target, package, or serving authority.  Keep
the revision-267 minimum 128 games/update, 64/64 learner seats, fixed 7,172
public total, H10 minimum, receipts, and dashboard reporting unchanged.

Under revision 231-MCTS, one stuck, timed-out, crashed, or malformed fleet
worker may never crash or cancel the whole r229 BO1000. Preserve an immutable
attempt receipt, release only that attempt's claim, and requeue the exact game
identity without double-credit. Healthy in-flight and queued games continue.
Track consecutive failures per execution host/slot and quarantine only the
failing capacity after three consecutive failures; a later explicit passing
host preflight may clear quarantine. Never retry a structurally invalid result
as valid evidence, lose a completed game, or broaden recovery into a managed
service or interactive-session restart. Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 230-MCTS, raise the complete ordered legal-action enumeration
ceiling for the r229 mirror from 4,096 to exactly 65,536. Apply the identical
ceiling to both the r228-MCTS and standard r195 no-MCTS arms. The observed
6,720-action prompt must be fully enumerated rather than downgraded to a direct
fallback, sampled, threshold-pruned, or treated as forced. The finite 65,536
ceiling remains fail-closed for larger pathological prompts, which must be
counted explicitly and may not silently influence the reported MCTS rate.
Typed source:
`state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`.

Under revision 228-MCTS, the active deliverable is the rough one-shot Kaggle
full-gameplay viability test before BO1000. It preserves r222 byte-for-byte.
At every branching gameplay decision, one submission process and one loaded
stock r195 `cg/libcg.so` DSO host eight persistent internal `AgentStart()`
simulator arenas / CPU lanes—not eight competition agents—searching one
master-owned shared MCTS tree. Call `SearchBegin` exactly once per lane per
branching decision and retain each exact `(lane, handle, SearchId)` tuple
through repeated depth waves. After each assigned simulator action, advance to
discover the next legal actions; batch up to eight returned frontier states
through the frozen r195 GPU evaluator; back every result into the sole tree;
and continue until a branch boundary, clean decision/turn deadline, or
convergence. A non-deadline searched action must be legal and have at least one
completed root backup; only forced single-action prompts may bypass search.
Unforceable randomness stops a private branch before its outcome: no guessing,
private sampling, or unobserved advance.

A clean per-decision/turn deadline is nonfatal only after all heads and
reservations are cleaned. With a legal fully backed root action, return the best
such action. With zero completed backups caused solely by that clean deadline,
use the frozen direct policy after cleanup. Zero backups from any non-deadline
or structural cause are a hard failure. Missing lane, crash, unclean deadline,
stale/duplicate lane state, illegal edge/action, incomplete wave, tree-invariant
violation, cleanup leak, or unbacked non-deadline action exits nonzero, with no
greedy, serial, or partial-MCTS fallback. Emit exactly one explicit success
marker only after the full gameplay loop passes. The primary signal is eight
head launch and sustainment in gameplay, not a claim that a deadline never
occurred. After a local exact-package full-game smoke, submit exactly one direct
diagnostic with the existing `DONT USE FOR REVIEW — 8-LANE SHARED-TREE
VIABILITY` label—no queue, retry, or copy. Typed contract:
`state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`.

Under revision 222, the Replay Model Inspector reconstructs only the selected
step and factorized stage on demand on Elmo's `cuda:0`; selecting a game no
longer queues every remaining step/stage behind the requested trace. The
browser gives the selected trace at most 20 seconds of visible wait, aborts
stale client requests, and retains only successfully requested traces in its
device-local memory cache. Submission `55410353`'s exact archived runtime and
checkpoint become the inspector's one resident GPU runtime so its cold and
warm selected traces do not spawn a fresh isolated Python/model load per step.
All exact-runtime, replay-digest, setup-short-circuit, and
`recomputed_not_historical` truth labels remain binding; this is never a claim
that Kaggle stored logits. Other package identities remain fail-closed and
isolated. Typed contract:
`state/replay-model-inspector-on-demand-gpu-trace-r222.json`.

Under revision 223, every header in the Matchup Adapter ON/OFF legal-action
comparison is a keyboard-accessible sort control: Legal action, Adapter ON
chance, Adapter OFF chance, signed Change caused by adapter, and Chosen. The
default preserves archived legal-option order; repeated clicks toggle stable
ascending/descending order, non-finite values remain last, and sorting never
mutates source probabilities, choices, or replay data. Typed contract:
`state/replay-model-inspector-adapter-comparison-sorting-r223.json`.

Under revision 214, run a separate testing-only BO1000 mirror of the exact
revision-195 NO-RTP Alakazam package against itself: the direct arm uses the
frozen package unchanged and the experimental arm wraps that same complete
frozen model in the existing public-history root-sampled `BeliefMCTS`. The
trained Matchup Adapter must be runtime-on for both arms and bind the exact
r195 public matchup tree
`sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049`.
The only runtime difference is BeliefMCTS action selection; the exact r195
checkpoint `sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
bundle `sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145`,
deck, full policy/value/fusion path, and adapter bank are the same. RTP,
legacy RTP sidecars/executors, the older guide-linear/guide-logit layer, and
Guide2Vec are off in both arms. Run exactly 1,000 games as 500 seeded
RNG/deck-order matched seat-swapped pairs: BeliefMCTS is each seat 500 times
and actual first/second 500 times. The one typed easy-to-change budget object
sets hard monotonic ceilings of 20 seconds per actual turn and 5 seconds per
atomic action, charging particle sampling, simulator, model, adapter,
validation, backup, and receipt work; a deadline or insufficient trusted
search falls back exactly to direct r195 policy. This is a simple
public-history root-sampled belief search, not revision 207's exact-chance
inter-turn simulator experiment: sampled hidden particles and coin outcomes
must be labelled `root_sampled_belief_mcts_non_r207_exact_chance`, never an
exact finite-chance expectation. Preserve r207 research unchanged. The report
must give paired/seat/first-second results, timing, simulations, leaf/node
counts, requested simulation-target completion, sampling/support data, and a
plain-English results table; a finite complete-tree rate is explicitly not
applicable to the stochastic root-sampled tree. No training, parameter update,
serving, selector, promotion, checkpoint publication, Kaggle, r175 restart,
or iteration-21 authority is granted. Typed contract:
`state/alakazam-simple-belief-mcts-bo1000-r214.json`.

Under revision 215, supersede revision 214's unlaunched per-decision simple
BeliefMCTS execution semantics with a separately versioned, testing-only
full-actual-turn BO1000. Preserve the r214 contract and all of its unlaunched
preflight evidence. The direct arm remains the exact revision-195 NO-RTP
Alakazam package; the experimental arm is that same frozen full model and
runtime path, including the runtime-on checksum-bound Matchup Adapter, wrapped
only in full-turn public-history root-sampled BeliefMCTS action selection. RTP,
legacy RTP, guide-linear/guide-logit, and Guide2Vec remain off in both arms.
The source-backed outer game clock is 600 seconds / 10 minutes with its
easy-to-change reserve and fair-share guard. Each actual turn receives the
minimum of its easy-to-change default 20-second planner pool, the current
outer-game allocator result, and remaining game time; it therefore shrinks
when the 600-second game clock is tight. Search starts at that turn's first
atomic decision. Later atomic steps do not reset either a 20-second pool or a
fresh five-second search allowance: their effective component allowance is the
minimum of the configured five-second model/simulator-operation ceiling,
remaining same-turn pool, and remaining game time. A deterministic cached
continuation charges only its required validation work. Receipts must bind the
clock configuration identity and report game time remaining, allocator result,
turn-pool default/effective value, work used, remaining-before/after, and each
effective operation allowance. At every hop, verify the fresh public observation fingerprint,
complete legal action order or exact factorized equivalent, and selected-action
legality before sending exactly one selected action to the real game. Private
simulator actions never reach the real game. If every fingerprint matches, use
the cached deterministic selected branch without re-evaluating or rebuilding;
rebuild only on a real divergence or explicit chance/information boundary and
only from the remaining same-turn pool. A new actual turn clears the tree and
cache. Within that turn tree, evaluate each unique deterministic model-input
state at most once and reuse cached frozen policy/value on repeat visits while
keeping node visit/value statistics separate. Real simulator deterministic
successor expansion and value backup, including multi-step depth when
available, are mandatory; a root-only reranker is not MCTS. A public-observation
match alone may never merge transpositions or share a model value. Merge an
`A→B` and `B→A` result only when the native simulator attests exact complete
semantic-state equality including hidden state, RNG state, pending effects,
selection/configuration, and future legal-action order. A valid native
`ActionsCommute` certificate may skip the second order; without that exact
identity or certificate, expand each order separately and fail closed. The
current packaged API exposes neither attestation, so r215 may not claim
transposition merges or model-evaluation savings; it must report attempted,
accepted, rejected and reason-coded merge telemetry. A small finite
chance point may be called exact only when the engine exposes all outcomes,
exact rational weights, and independently advanceable children; otherwise
sample privately, label it sampled/opaque, and rebuild on the realized
boundary. This remains `root_sampled_belief_mcts_non_r207_exact_chance`, never
revision-207 exact chance. There is no fixed 50-simulation target or target
completion gate: one valid simulation is the minimum, a very high emergency
safety ceiling is only a stop guard, and no valid simulation/deadline falls
back exactly to direct r195 policy. The BO1000 remains 500 seeded
RNG/deck-order matched seat-swapped pairs with exact 500/500 seat and actual
first/second balance, training-ineligible, and shadow-only. The report must
include the dynamic timing pool/allowance values, actual simulations, unique
states, cache proof, simulator successor/terminal/backup/multistep telemetry,
branch-cache/rebuild reasons, chance labels, paired results, and a plain-English
table. No training, parameter update, serving, selector, promotion, checkpoint
publication, Kaggle, r175 restart, or iteration-21 authority is granted. Typed
contract: `state/alakazam-full-turn-belief-mcts-bo1000-r215.json`.

Under revision 216, authorize the requested local exploratory BO1000 now as a
separately versioned relaxation of r215's advanced launch proofs only. Preserve
r215 byte-for-byte and retain it as the contract for exact/promotion-qualified
work. The direct arm and the approximate BeliefMCTS arm both remain the same
frozen r195 NO-RTP package, checkpoint, deck, full model path, and runtime-on
Matchup Adapter/tree; RTP, legacy RTP, guide-linear, guide-logit, and Guide2Vec
remain off in both arms. The local run may use the existing approximate
BeliefMCTS/search and available turn-cache APIs without first proving perfect
native complete semantic-state equality or an `ActionsCommute` certificate.
It must use one shared dynamic actual-turn pool from the 600-second outer game
clock: `min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))`. A healthy
600-second game therefore receives the full 20 seconds and shrinking starts
only below 190 seconds; the 5-second component-operation ceiling is within the
same residual turn pool. This explicitly supersedes r215's stale 600/30/64
fair-share allocation for r216 only, with direct-policy fallback. Run all 1,000 games as 500 seeded
RNG/deck-order matched seat-swapped pairs with exact 500/500 MCTS seat and
actual-first/second balance. Label every result
`local_approximate_belief_mcts_non_exact`,
`root_sampled_belief_mcts_non_r207_exact_chance`, and
`non_promotion_exploratory_result`; it may not claim native exact-state,
commutation, transposition-saving, or r207 exact-chance proof. This is local
testing only: no training, parameter update, serving, selector, promotion,
checkpoint publication, Kaggle API call, queue, upload, or submission, r175
restart, or iteration-21 authority is granted. Typed contract:
`state/alakazam-local-approximate-belief-mcts-bo1000-r216.json`.

Under revision 217, clarify the separately managed revision-212 Guide2Vec
experiment before training or evaluation starts.  The frozen revision-195
Matchup Adapter bank and exact public tree remain runtime-on, identical, and
frozen during training-latent extraction and in both later BO1000 arms.  The
candidate runtime contains exactly one frozen Guide2Vec component.  The direct
control runtime contains no Guide2Vec object at all: zero modules, parameters,
state keys, forward hooks, or linear transforms; a disabled or zero-weight
Guide2Vec component is not an acceptable control.  Historical guide-linear and
guide-logit layers remain absent.  Per-game graph-absence, graph-difference,
and adapter-parity receipts are mandatory.  This clarification changes no
other BO1000, grants no production or Kaggle authority, and remains owned by
`state/alakazam-guide2vec-no-mcts-bo1000-r212.json`.

Under revision 218, supersede only revision 216's local approximate BO1000
execution semantics for a new separately versioned run; preserve every r216
byte and any historical r216 authorization or receipt unchanged. The r218
experimental arm may launch approximate BeliefMCTS search only at the first
actual decision of an actual turn, for at most
`min(10.0, dynamic_game_allowance)` seconds, where the dynamic allowance remains
`min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))` from the
600-second outer game clock. The full first-decision search-or-fallback operation
is capped at that ten-second allowance: at a full allowance, reserve 9.5 seconds
for private search and 0.5 seconds for the exact direct-policy fallback.
Individual model/simulator calls remain observed and telemetrized but have no
inherited five-second outer call cap. Later same-turn decisions may
only execute a fingerprint-validated cached plan or the exact frozen r195
direct-policy fallback; they may not launch a fresh search, rebuild a tree, or
open a new search allowance. A missing, invalid, diverged, or exhausted cache
falls back directly rather than searching again. There is no fixed simulation
or depth target or completion gate—only the high emergency safety guard. An
early search stop is truthful only with an explicit stable-root convergence
receipt and a fully backed-up legal selected action; elapsed time, simulation
count, or a partial/unbacked action cannot cause early stop. The 1,000-game
matched BO1000 itself still completes all games without an early best-of stop.
Both arms remain the exact frozen r195 NO-RTP package with the identical
runtime-on Matchup Adapter, and all non-exact/non-r207/non-promotion labels
remain required. A future Kaggle runtime, if separately authorized by the
owner, should target an AWS `p5.4xlarge`-equivalent H100 80GB / 256 GiB / 16
vCPU environment with batched frozen inference and resource-aware search; this
record grants no Kaggle API, queue, upload, submission, runtime, service, RTP,
selector, serving, promotion, training, or launch action now. Typed contract:
`state/alakazam-local-first-decision-belief-mcts-bo1000-r218.json`.

Under revision 219, correct the first-decision-only restriction for the next
separately versioned local approximate BO1000 while preserving revision 218
byte-for-byte. The r219 experimental arm uses one source-backed shared
45-second planner pool per actual turn, dynamically bounded by
`min(45.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))` from the
600-second game clock. Every meaningful search segment—including the first
one—may use at most 15 seconds and only remaining pool; there is no
first-search-only restriction or special lower first-search cap. A later
search is permitted only at a meaningful still-active-turn boundary: a
non-forced legal decision, realized chance/information divergence, or a
validated cached-plan endpoint. Valid deterministic cached or obvious/forced
steps consume validation/dispatch time only and do not force a search. An
actual turn end closes and discards its pool/cache; a simulated turn end is a
leaf/terminal evaluation within the current segment, and the next real turn
gets a fresh pool. If time is insufficient or a search result is untrusted,
execute exact frozen r195 direct policy for that current legal decision without
a hard abort.

For a fully exposed finite chance point—such as a two-outcome coin flip, a
six-outcome standard die, or any complete distribution of at most six outcomes
with exact probabilities, successors, and future legality—enumerate every
outcome, back up its exact probability-weighted value, and continue evaluation
beyond its children within the current segment/turn budget. Hidden,
incomplete, stateful, complex, unbounded, or unforceable randomness remains a
realized boundary and re-root/cache-validation point. A finite-chance node
claims exactness only with a force/enumeration/probability receipt; the whole
local run remains non-r207 exact chance. PUCT must prioritize higher frozen
policy-prior lines naturally while preserving every positive-prior legal line;
do not use an arbitrary probability threshold to prune it. Positive-probability
finite-chance children receive bounded coverage when valid and budget permits.
There is no fixed simulation/depth target beyond the emergency guard, and
early stopping still requires explicit stable-root convergence and a
fully-backed-up legal selected action.

After fresh r219 preflight, run a 10-game/5-seeded-pair r219 canary before
BO1000. It must report total MCTS turns; one-search-segment turns; turns with
one or more later re-searches; mean/max segments per turn; cache-only later
steps; chance enumeration/rebuilds; simulations, depth, convergence,
fallbacks, and MCTS action changes. Only a valid canary permits the complete
500-pair/1,000-game mirror. Both arms remain the same frozen r195 NO-RTP
package with runtime-on Matchup Adapter and guide-linear, guide-logit,
Guide2Vec, RTP, and legacy RTP off. This remains local-only,
training-ineligible, non-exact, and non-promotion; it grants no training,
serving, selector, checkpoint publication, Kaggle, or legacy-RTP authority.
Typed contract: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json`.

Under revision 183, direct local access through
`http://bert:8780/replay-inspector/` must recognize Tailscale's IPv4
shared-address block (`100.64.0.0/10`) as the dashboard's established private
overlay. The authorization decision still uses the actual socket peer rather
than request headers; public and adjacent address ranges remain rejected; and
the existing local Host/Origin checks, GET-only fixed-prefix proxy, credential
stripping, bounded response, loopback tunnel, and independent inspector
service boundaries remain unchanged. Typed contract:
`state/replay-model-inspector-tailscale-local-gateway-r183.json`.

Under revision 184, Kaggle submission `55217604` is a permanent explicit
replay-sync special case even though it predates the recurring sync's unchanged
minimum discovery ID. The ordinary hourly timer, discovery floor, and automatic
discovery of new owner submissions remain unchanged; the special ID is unioned
into the default discovered set and is rechecked on every ordinary hourly run
for newly available games. Its replay index remains browseable independently
of model-analysis provenance, and model analysis stays unavailable until its
own exact checkpoint, bundle, submitted runtime, matchup tree, replay digests,
and parity receipt are present and verified. It must never inherit the newer
r175 Alakazam artifacts. Typed contract:
`state/replay-model-inspector-submission-55217604-special-case-r184.json`.

Under revision 185, Matchup Adapters are required on for future Kaggle
submissions whenever the submitted checkpoint carries a trained adapter bank.
Packaging must fail closed unless it includes a checksum-verified,
runtime-enabled matchup tree and exact submitted entry point that activates the
bank; installed or nonzero weights alone are not activation. For already
submitted packages, the Replay Model Inspector must reproduce the exact
checksum-bound startup behavior: a verified packaged tree enables the bank for
the request, an accepted routable route is reported active, and an unknown or
unroutable route is reported as an exact bypass. The cached model's serialized
dormant flag is training-safety state and must not be mistaken for the submitted
serving state. Request-scoped reproduction must restore shared cached state,
and historical traces remain labelled causal re-evaluations rather than
recorded logits. Typed contract:
`state/replay-model-inspector-submitted-adapter-runtime-r185.json`.

Under revision 186, every Replay Model Inspector submission choice and
selection summary must retain the exact numeric submission ID even when a
human-readable label is available. It must also show the submission's cached-
replay win rate with an explicit wins/eligible-games denominator, computed only
from the acting submission seat's source-backed archived outcomes. Missing or
malformed outcomes are excluded and reported; they are never treated as wins
or losses, and the UI must say unavailable when no eligible outcomes exist.
Typed contract:
`state/replay-model-inspector-submission-win-rate-r186.json`.

Under revision 187, the Replay Model Inspector adds two explicitly distinct
playground views. The instant Decision Influence view gives every architecture-
present fused head a bounded `0x`--`2x` scale (`1x` exact baseline), recomputes
the real nonlinear fusion path for only the selected causal decision, and shows
baseline versus counterfactual legal-action logits, probabilities, and choice.
It never approximates by summing leave-one-out effects and never calls these
scales training weights. A separate Training Weight Recipe view displays the
source-backed learning-loss multipliers such as current-deck guide weight
`0.05` and strategic-head loss weights; changing those cannot alter an existing
forward pass and requires a later isolated fine-tune to change policy behavior.
Both views remain ephemeral; they never edit checkpoint tensors, replay bytes,
training data, the active selector, managed training, or a canonical recipe.
Kaggle/evaluation replays remain training-ineligible. Typed contract:
`state/replay-model-inspector-head-influence-playground-r187.json`.

Under revision 188, the Replay Model Inspector's decision selector and public
decision-step API show only turns acted by our submitted agent,
`Challengestone`. The authoritative filter is the replay archive's
submission-bound own-agent seat, not a fixed numeric seat or an unverified name
match, because Challengestone may occupy either seat. A selectable row must
also be `ACTIVE` and its masked observation's integer `current.yourIndex` must
equal that archived seat; stale `select` data on inactive rows is never a
decision. Opponent events remain
available only as hidden causal history needed to reconstruct Challengestone's
later decisions; they are not selectable, and a direct opponent-step trace
request fails closed. If the archived own-agent seat is absent or ambiguous,
the decision list and trace are unavailable with an explicit reason rather
than exposing or guessing a side. Typed contract:
`state/replay-model-inspector-own-seat-decisions-r188.json`.

Under revision 189, every user-facing Replay Model Inspector submission ID is
rendered as its exact base-10 identifier string, such as `55315274`. Submission
IDs must never pass through generic numeric metric formatting, scientific
notation, rounding, digit grouping, or abbreviation. Internal catalog lookup
may remain integer-based, but the public payload supplies an exact decimal-text
field and the selector, selected-submission summary, game context, and trace
context use that text. Typed contract:
`state/replay-model-inspector-submission-id-text-r189.json`.

Under revision 190, the Replay Model Inspector game chooser has a mobile-
friendly search field that filters the already loaded submission games by full
or partial exact-decimal game ID and by available player name. The native game
select remains the final chooser, displays only the matching games, reports the
match count or an explicit no-match state, and safely selects the first match
when the prior selection is excluded. Changing submissions clears the filter.
This is presentation-only: it never changes the replay archive, provenance,
decision visibility, or server-side selection authority. Typed contract:
`state/replay-model-inspector-game-filter-r190.json`.

Under revision 191, every trace-ready decision with a checksum-verified,
actually applied Matchup Adapter route shows an explicit Adapter ON versus
Adapter OFF comparison. Adapter ON is the exact submitted-runtime policy for
that decision; Adapter OFF is a second exact forward result with only the
decision's adapter route bypassed while all other causal inputs stay fixed.
The view shows both chosen actions, each legal action's probability under both
states, and the signed probability change caused by the adapter. It is a
causal re-evaluation, not recorded telemetry. An unknown/unroutable route or an
unattested submitted runtime has no truthful force-on counterfactual and must
remain explicitly unavailable rather than inventing a route. The comparison
never changes checkpoint tensors, replay bytes, training state, or serving
authority. Typed contract:
`state/replay-model-inspector-matchup-adapter-counterfactual-r191.json`.

Matchup Router Format 6 is active in the Teal Mask Ogerpon ex lineage. It uses 64 fixed
physical slots so ordinary archetype additions, retirements, renames, and
priority changes do not change tensor shapes or checkpoint format. Its
compatibility receipts passed at the Teal receipt-backed activation boundary;
older Router Format 5 checkpoints remain immutable historical lineage.
This router-format number is independent of the training-core revision and
accepted-policy generation. The previous checksum-accepted production fallback
is Accepted Policy Generation 9
(`sha256:7d9b60e68f4c51bb931298ae3941e5b7bddf1370566b23d18acadd33e8357056`).
Accepted Policy Generation 15 is reserved for the revision-118
checksum-bound Marnie latent-policy activation at the restarted iteration-6
boundary; generations/attempts 10 through 14 remain immutable history and are
never renumbered or overwritten. Until the revision-118 activation receipt is
published, generation 9 remains the rollback parent. After publication,
generation 15 is the active Marnie learner and generation 9 remains its
preserved fallback. The post-Thwackey Policy Generation 10 candidate was attempted and immutably
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
**Accepted Policy Generation 15** is the owner-authorized Marnie latent-policy
boundary generation once its checksum-bound activation receipt is published;
Accepted Policy Generation 9 remains the immutable rollback parent. Policy
attempts 10 through 14 remain historical and are not reused. Bare `Vn` labels
are not permitted for any of these current-status surfaces.

Revision 124 restarts the uncommitted Marnie iteration 6 with the repaired
self-play scheduler. Elmo keeps all 36 and Bert all 16 execution workers fed
through one exact game per request socket; neither endpoint may hide a private
multi-game batch. Only when exactly 20 shared jobs remain unclaimed does Bert
receive no new self-play claims and its controller-owned queued games return to
the shared Blackwell pool; Elmo retains one executing wave and has first remote
claim on the remaining self-play tail. This allocation change does not apply to
public mix, research control, promotion, or formal holdout.
The frozen Dragapult-family public opponents already available as executable
models—Dragapult/Dusknoir and Hammer Pult—remain explicitly included in the
replay-eligible public-practice roster. This opponent restoration does not
restore plain Dragapult, Dragapult/Blaziken, or Dragapult/Dudunsparce as
specialist-training targets.

Revision 125 forbids redundant remote checkpoint loading or full SMB payload
rehashing when the weights are already resident under a complete checksum-exact
health proof. Both the
production boundary and remote canary reuse the resident weights only when the
controller, every live leaf, the freshly advertised primary identity, and the
pinned digest all agree and are healthy. Production fails closed to the
ordinary verified reload for a missing field, unhealthy leaf, identity
mismatch, or genuinely new digest. A diagnostic canary instead refuses an
implicit reload and requires an explicit `--force-reload`. A new checkpoint
therefore loads exactly once per production endpoint; repeated checks of
unchanged healthy weights seed the digest-addressed process staging cache and do
not incur another multi-minute load or network rescan.

Revision 126 completes the runtime-inert implementation contract for revision
120: exact observed-list validation and clustering, family-macro collection and
replay provenance, capability-masked existing-head loss aggregation, isolated
two-round antithetic SPSA evidence, checksum-exact iteration-9 upload trigger
validation, an idempotent post-commit pause hook, atomic managed activation,
rollback scheduling, and owner-only future package switching.  No observed
activation-ready manifest, successful iteration-9 upload trigger, passing
shadow study, pause receipt, migration receipt, or selector change is asserted
by this implementation revision.  Their absence keeps the exact-list recipe
active and all new paths runtime-inert.

Revision 127 closes the two verified causes of false Elmo reservation without
execution at the stopped iteration-6 recovery boundary. Historical
content-addressed checkpoint objects are verified by SHA-256 beside TrueNAS
storage through the root-bounded `checkpoint_digest_verify_v1` worker
capability; persistent receipts and an exact local-file identity cache prevent
repeat SMB payload reads and repeat local checkpoint hashing. Initial remote
request sockets are opened concurrently across Elmo and Bert with bounded LAN
connect/hello deadlines, so a slow Bert SYN cannot keep a fully ready Elmo at
zero admitted jobs. Activation is bound to
`state/remote_scheduler_checkpoint_activation_r127.json`: all 50 clone sockets
connected in `0.199` seconds with zero failures, Elmo admitted/executed `36`,
Bert admitted/executed `16`, the dashboard resolved the live managed PID with
all required sources current, and iteration 6 advanced under the unchanged
96-worker Blackwell, 36-worker Elmo, and 16-worker Bert profile.

Revision 128 records the post-resume runtime without changing its scheduling
contract. Marnie iteration 6 completed its 1,024-game self-play phase and
advanced into public mix under managed PID `1267326`, Accepted Policy
Generation 15 checkpoint
`sha256:9ff2a8bcaf9ee51db1f6bb7dd86fb5d480a1737d80859140323f9b6fcab36cc5`,
and runtime registry
`sha256:72b61af5e788b71ef74caa8922f3ce4d2cca51fca11170de4b34b4e348381814`.
The post-resume snapshot is bound by
`state/marnie_iteration6_resume_snapshot_r128.json`. The 96+36+16 fleet,
one-game self-play request ownership, and exact 20-unclaimed-game self-play
tail rule remain unchanged.

Revision 129 closes Crustle's v2-corpus portion of the post-Marnie pre-stage
without granting Crustle runtime authority. The checksum-bound imported corpus
contains 26,932 exact games, 1,428,142 decisions, and 33,620 guide rows under
`crustle-north-star-v2`; its ready receipt, protected pointer, manifest, and
final validation receipt all match the import receipt. The dormant five-unit
Crustle chain remains loaded and inactive, and the new handoff receipt is
`/home/inzi/poke-bot-agent/outputs/state/post-marnie-crustle-r113-handoff-ready-v2.json`.
Crustle still cannot bootstrap until Marnie's immutable iteration-20
completion, and its register step must resolve and checksum-bind the
then-current H10 runtime registry rather than silently inheriting a stale
pre-Generation-15 registry.

Revision 130 makes the Marnie new-system transition after the iteration-9
Kaggle submission immutable. A successful checksum-exact iteration-9 upload is
the final old-system boundary: the first later Marnie training collection must
use the new system. No controller, fallback, missing-study path, or later
metadata edit may start another old-system collection after that successful
upload. All new-system manifests, studies, migration inputs, and service
preflight must therefore be prepared before the trigger. If any required
activation evidence is incomplete when the upload succeeds, fail closed by
pausing before the next collection; do not weaken the new-system gates and do
not silently continue the exact-list/old-system recipe. The currently active
iteration remains unchanged.

Revision 131 reconciles the resumed controller and arms the already-staged
revision-130 hook without changing the live iteration or scheduler. The
one-shot managed boundary service waits for the immutable iteration-6 commit,
then restarts only the managed Marnie trainer before iteration-7 collection so
the running process inherits the fail-closed iteration-9 upload hook. The
iteration-9 submission remains the immutable final old-system boundary; this
operational restart does not activate the new system early. The 96-worker
Blackwell, 36-worker Elmo, 16-worker Bert allocation and exact 20-unclaimed-game
self-play tail rule remain unchanged. The armed-state receipt is
`state/marnie_new_system_hook_monitor_armed_r131.json`; the activation service
must publish `/home/inzi/poke-bot-agent/outputs/state/marnie-new-system-hook-activation-r130.json`
after it verifies the new PID and both boundary-hook environment paths.

Revision 132 records the owner's immutable reaffirmation of the Marnie
transition boundary. Marnie must train on the current system through iteration
9 and through the successful checksum-exact iteration-9 Kaggle submission.
Preparation, shadow study, hook arming, or an otherwise-ready new-system
artifact never authorizes early activation. Only after that successful upload
may Marnie begin training on the new system, starting with the first later
collection. This order is not changeable by a controller, fallback, gate
interpretation, or metadata reconciliation. The active iteration, scheduler,
fleet allocation, and self-play tail behavior remain unchanged.

Revision 133 records the receipt-backed completion of the revision-131
operational hook. Iteration 6 committed at checkpoint
`sha256:516ce12b1a2e984de62da9ea65cb52611ed0eceee40433d4de248b7f1a28472e`;
the one-shot service then restarted only the managed Marnie trainer between
iteration 6 and iteration 7. The new process loaded the two fail-closed
iteration-9 boundary paths, while the archetype-family sampler and typed loss
vector remain absent from its environment and therefore runtime-inert.
Iteration 7 began on the current system. The 96+36+16 fleet, one-game socket
ownership, and exact 20-unclaimed-game self-play tail rule are unchanged.
Receipt: `state/marnie_new_system_hook_activation_r130.json`, byte-identical to
the managed receipt at
`/home/inzi/poke-bot-agent/outputs/state/marnie-new-system-hook-activation-r130.json`.

Revision 134 fixes the first post-upload Marnie training sequence. After the
successful checksum-exact iteration-9 Kaggle upload and atomic new-system
activation, hot-start the exact uploaded iteration-9 learner and run exactly 25
expert bootstrap epochs over the checksum-pinned active Marnie's Grimmsnarl ex
corpus (73,082 acting-seat game records and 6,828,373 decisions). Package and
submit that exact bootstrap checkpoint to Kaggle, and do not begin the first
new-system self-play collection until the bootstrap submission has a successful
checksum-exact upload receipt. A failed or pending bootstrap/upload pauses the
new-system chain without reverting to an old-system collection. The currently
running iteration, scheduler, 96+36+16 fleet, and exact 20-unclaimed-game
self-play tail rule remain unchanged.

Revision 135 explicitly authorizes one bounded leaderboard-provenance exception
for that post-upload bootstrap. Public expert replay acting seats whose raw
`TeamNames[seat]` exactly match a team in a checksum-pinned PTCGReplay top-100
snapshot receive sample-size-tiered training importance from the same pinned
Grimmsnarl corpus: `1.5x` for 1--31 acting-seat games, `2.0x` for 32--127,
`3.0x` for 128--511, and `4.0x` for 512 or more. Every unmatched or
unverifiable seat remains `1.0x`. The join key is exact
`episode_id + seat + team_name`; the snapshot and derived weight index are
immutable inputs, and the receipt reports matched/unmatched counts, per-team
support, tier counts, and effective weight mass. This changes sample importance
only: replay actions and causal labels are never rewritten, Kaggle evaluation
replays remain training-ineligible, and no live pre-iteration-9 training or
scheduler behavior changes.

Revision 136 supersedes only revision 135's upper support bands after verifying
the pinned training split contains 22 top-100 pilots with at least 512 games,
nine with at least 1,024, and four with at least 2,048. Preserve the lower
tiers, use `4.0x` for 512--1,023 acting-seat games, raise 1,024--2,047 to
`5.0x`, and cap 2,048 or more at `6.0x`. Validation remains unweighted and all
unmatched or unverifiable seats remain `1.0x`. This remains runtime-inert until
the revision-134 post-iteration-9 bootstrap and does not alter active Marnie
training, collection, evaluation, scheduler, fleet, or tail behavior.

Revision 137 raises only the two highest same-training-split support bands in
response to the owner's explicit higher-bound decision. Preserve every lower
tier, use `6.0x` for 1,024--2,047 acting-seat games, and use a bounded `8.0x`
for 2,048 or more. The pinned split contains nine top-100 pilots in the former
band and four in the latter, so sparse pilots receive no additional
amplification. Validation remains unweighted, unmatched or unverifiable seats
remain `1.0x`, and the change remains runtime-inert until the revision-134
post-iteration-9 expert bootstrap.

Revision 138 raises those same evidence-backed bands one bounded step further:
`7.0x` for 1,024--2,047 acting-seat games and `10.0x` for 2,048 or more.
The lower tiers remain unchanged. The four pilots in the highest band each
have 2,094--2,539 exact training games, so the higher ceiling is restricted to
large-sample pilots; validation remains unweighted and all unmatched or
unverifiable seats remain `1.0x`. This change applies only to the exact
25-epoch post-iteration-9 expert bootstrap.

Current runtime reconciliation: iteration 9 committed, the exact gate restored
the stronger iteration-7 learner checkpoint
`sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381`,
and Kaggle submission `55230247` uploaded that exact learner. The managed
family study then sealed
`failed_closed_inconclusive_after_two_rounds`; revision 139 supplied distinct
owner-ceiling authority without relabelling that measured result. Exact round-1
`plus` was sealed in request
`sha256:66717a6c2060a1a38227c35ea903ddea4c793a395cf3aeaa9dc187d322df2f75`,
and the family sampler plus typed loss vector activated atomically. The exact
revision-138 weighted, guide-off 25-epoch bootstrap subsequently completed and
selected checkpoint
`sha256:fe8a4a1ab94433e5f32cfcc194effdf0ce4f49745fd4bb26f1f3a20f7ecc2ae8`.
Kaggle submission `55248489` uploaded that exact checkpoint successfully.
The managed Marnie service now resumes at the iteration-10 boundary from that
checkpoint under Accepted Policy Generation 15, the 48-variant family sampler,
19-head Fusion-v3, and guide weight `0.0`. A new exact heldout gate is required;
no evidence from the superseded checkpoint transfers to this learner. Outcome
receipt: `state/marnie_postupload_family_study_outcome_r138.json`; bootstrap,
upload, activation, and evidence-repair receipts remain checksum-bound on the
training host. The post-activation status-78 monitor/resolution timer remains
mandatory at the first new-system iteration-10 commit.

Revision 139 records the owner's explicit decision to activate the family
system because it is the purpose of this transition, despite the valid
two-round shadow study remaining statistically inconclusive. Preserve
`study.json` and every negative lower-bound result byte-for-byte and never call
the measured study a pass. Owner ceiling authority selects round 1 `plus`, the
only tested direction with a positive overall point estimate: increase
`core_setup_continuity`, decrease `long_horizon_prize_pressure`, and increase
`resource_attack_readiness` under its exact tested vector. Materialize a
separate checksum-bound owner-ceiling receipt, selected loss vector, candidate
registry, and activation request; validate the same parent, isolation,
causality, manifest, loss-contract, sealed-state, and atomic-migration
invariants as an ordinary passing activation. Then activate the family sampler
and typed loss vector atomically, run the exact revision-138 weighted 25-epoch
bootstrap, submit epoch 25, and keep iteration-10 collection forbidden until
that upload succeeds. The unchanged post-activation status-78 monitor remains
mandatory and must restore the exact-list parent path if its own gate fails.

Revision 140 permanently retires the Marnie current-deck guide for this
lineage. Its multiplier is exactly `0.0` for the exact 25-epoch bootstrap and
every later Marnie RL or rehearsal update; guide target generation,
guide-conditioned losses, and guide action influence are disabled. The guide
contract, the failed zero-label epoch-1 attempt, and any derived labeled corpus
remain immutable audit evidence only. This retirement does not disable or
reduce the archetype-family sampler, Accepted Policy Generation 15 latent
lookahead, policy/value learning, strategic heads, setup/combo heads, decision
fusion, or matchup adapters. Resume the same checksum-exact 73,082-game,
6,828,373-decision weighted bootstrap from the uploaded iteration-9 parent and
retain the unchanged epoch-25 checksum-exact upload gate before iteration 10.
The corrected r140 submission/activation path is bound by
`state/marnie_guide_retirement_submission_chain_r140.json`; it requires the
zero-guide ready receipt and immutable retirement receipt before queueing or
activating, and the iteration-10 launch omits all guide target/curriculum
arguments while preserving the 19-head Fusion-v3 path.

Revision 141 clarifies that retirement means **non-authority**, not artifact
deletion. The checksum-bound Marnie guide may remain available for optional
offline shadow diagnostics, but live guide target generation stays disabled
and its effective loss weight remains exactly `0.0`. Shadow outputs never
receive gradients, contribute loss, enter Fusion-v3, alter action logits or
action selection, serve at runtime, select a checkpoint, affect promotion or
family-monitor gates, authorize self-play or submission, or block any phase.
Missing, invalid, unlabeled, or failed shadow evidence is recorded as shadow
unavailable and the authoritative non-guide path continues. Kaggle and formal
evaluation replays remain evaluation-only. This clarification does not restart
or otherwise alter the active 25-epoch bootstrap and is bound by
`state/marnie_guide_shadow_non_authority_r141.json`.

Runtime reconciliation under the same revision preserves the completed,
checksum-exact guide-off epoch-1 checkpoint after its legacy validator omitted
the newly active family-residual gradient heads. The repaired validator checks
the full effective expanded-head weight vector and resumes from that immutable
checkpoint without overwriting or retraining epoch 1. Receipt:
`state/marnie_postupload_epoch1_recovery_r141.json`.
The epoch-25 submission and iteration-10 activation services require that
recovery identity end-to-end under
`state/marnie_epoch_recovery_submission_chain_r141.json`; neither may discard
it while packaging, uploading, or switching the learner.
The live dashboard must render this third state explicitly as shadow-only,
weight zero, and nonblocking rather than collapsing it to either active or
absent; deployment evidence is
`state/marnie_guide_shadow_dashboard_r141.json`.

Revision 142 repairs the dormant iteration-10 next-start registry after the
already-activated family drop-in was found to override the revision-140
guide-retired registry with its older `0.05` family-study parent. The active
25-epoch bootstrap remains uninterrupted and guide-off. A new immutable
registry derivative preserves every family sampler, typed-loss, latent-policy,
Fusion-v3, matchup, and other non-guide field while enforcing guide weight
`0.0`, optional shadow-only availability, and zero runtime or blocking
authority. A lexically final managed drop-in selects that merged registry;
epoch-25 activation now fails closed unless its checksum-bound runtime receipt
is valid. The status-78 resolver keeps the merged guide-off family registry on
a monitor pass and selects the existing guide-off pre-family registry on a
family rollback, so neither branch can restore guide authority. Receipt:
`state/marnie_family_guide_shadow_runtime_r142.json`.

Revision 143 repairs the dormant Marnie-to-Crustle terminal handoff without
interrupting the active weighted bootstrap.  The effective Marnie completion
service had inherited an older drop-in whose `ExecStart` relaunched Marnie
instead of executing the completion transaction, and the Crustle register unit
still copied the pre-Generation-15 revision-113 registry.  A lexically final
completion overlay now runs the real checksum-bound completion transaction and
resolves the exact registry selected by the managed Marnie service at handoff.
The immutable Marnie completion receipt binds that registry path and digest;
Crustle registration must consume that binding, require guide weight zero and
Fusion-v3 runtime authority on the completed Marnie source, and remove every
Marnie-only family/guide runtime field before constructing the Crustle runtime.
The five Crustle units remain dormant and receive no selector, training, or
gradient authority before Marnie's immutable iteration-20 completion.  Receipt:
`state/post_marnie_crustle_runtime_rebind_r143.json`.

Revision 144 records the completed post-upload activation and reaffirms the
owner's guide-shadow boundary. The exact epoch-25 checkpoint is
`sha256:fe8a4a1ab94433e5f32cfcc194effdf0ce4f49745fd4bb26f1f3a20f7ecc2ae8`;
its ready, successful Kaggle upload (`55248489`), and activation receipts are
`sha256:466c2e039692c22f227b75fcbe5a36173ef7207c0d4082ee75057edfddbd246d`,
`sha256:f785d54b2578ad7a5446f4ed1278e85a19becc6b36f0ef9ad4bcecbf800f8258`,
and `sha256:60d0fbd4277be52f32e2f2f34517dcd6450ca51721e2f078bff311867253451e`.
The iteration-10 design migration activates the checksum-bound 48-variant
family sampler and guide-retired registry. The Marnie guide artifact may remain
available for offline shadow telemetry only: effective weight and target
generation are zero/off, it never enters Fusion-v3 or action choice, and its
absence, failure, or result cannot stop collection, promotion, submission,
monitoring, or handoff. The activated epoch-25 checkpoint receives a fresh
exact heldout evaluation; revision-153 repair receipt
`sha256:8af9603909fa79af9b0b672f51b1faabd80cb6ef91c4bc8f36f94eece0eef377`
proves that the superseded checkpoint's heldout evidence was cleared rather
than inherited.

Revision 145 repairs a finite-exact-bypass defect in Marnie's retired-guide
expert-rehearsal path. Packed option padding uses negative-infinity logits, so
the former inactive-loss expression `logits.sum() * 0` produced NaN loss
telemetry even though its derivative and all enabled non-guide gradients stayed
finite. Inactive guide, guide-curriculum, and setup placeholders now anchor to
the finite causal policy/value state and therefore remain exactly zero and
zero-gradient. The canonical tree and active deployment carry the repaired
module checksum, while the already-running iteration-10 rehearsal remains
uninterrupted because direct reproduction proved its real gradients finite.
The repair becomes executable at the next managed interpreter start on or
after a clean durable boundary. It does not restore guide targets, gradients,
fusion/action authority, gates, submission authority, or blocking behavior.
Receipt: `state/marnie_guide_shadow_finite_bypass_r145.json`
(`sha256:4ddc5f2e8a449a880e0c3e43a5e808f0ddac1dc8dff27be2c45def7c44373b15`).

Revision 146 repairs the iteration-10 RL candidate-save boundary after the
completed matchup-adapter fit exposed a guard mismatch: the RL path treated
legacy guide-off mode as an active strategic curriculum mode, then invoked a
serializer that correctly rejects legacy mode. Rehearsal and RL now share one
guard helper which returns no strategic-guide record for legacy/guide-off
lineages and preserves the existing record for actual strategic modes. The
managed service resumed from the immutable iteration-10 collection and
rehearsal receipts without recollection or rehearsal retraining. Marnie's
guide stays weight `0.0`, action-inert, gradient-inert, and nonblocking.
Receipt: `state/marnie_guide_off_rl_save_recovery_r146.json`
(`sha256:aa032e7f2e8e512c2c712bf1d7d72c166899545fe6d54c3f472fe901e56a12dc`).

Revision 147 makes Marnie's family replay draw immutable across recovery-only
source migrations. The family macro sampler must use the sealed collection's
`design_fingerprint_at_collection`, not the current source-tree fingerprint,
whenever a completed collection is being recovered. Iteration 10 therefore
restores its original 16,724-sequence replay projection while retaining the
independent exact 4,096/4,096 assigned and retained source-game split. No seat
receipt, collection, rehearsal, guide authority, or gate is rewritten or
weakened. The managed recovery passed the immutable seat-receipt boundary and
continued into baseline preparation under the guide-shadow-only runtime.
Receipt: `state/marnie_family_replay_seed_recovery_r147.json`
(`sha256:4379636ca9e012dab0fbaea0fb7d4f00b1f097a9228f60f5f3282bf327b09824`).

Revision 158 repairs a bounded public-mix exact-retention failure without
weakening the exact collection contract or interrupting the recovered Marnie
iteration 16. The first attempt completed all 1,024 self-play jobs and all
7,168 planned public jobs, but only 7,167 public records remained usable after
the historical four disjoint replacement seeds. It therefore failed closed at
8,191/8,192 and quarantined the uncommitted shard. Public-mix recovery now
retains the historical four seed lanes, adds 28 deterministic high-namespace
lanes, preserves the missing cell's exact seat/opponent/archetype/training-group
contract, promotes at most one usable record per missing primary cell, and
still fails closed if the exact 8,192 records cannot be proven. Self-play tail
allocation, the 96+36+16 fleet, public-mix weights, and every gate remain
unchanged. The repaired source is staged in the active deployment but does not
alter the already-running interpreter; it becomes executable only at a later
managed interpreter start. Receipt:
`state/marnie_public_mix_exact_retention_recovery_r158.json`
(`sha256:360a077e85bfa9f82cfa6f04ea1b9340c9ef88670867ac4999a99f4a948f445b`).

Revision 159 repairs the active-refresh handoff card after its generic
historical fallback again projected Slowking and an ordinary unfinished-roster
count beside the final-format Marnie learner. When the live runtime is the
separately versioned Marnie refresh, the handoff projection now explicitly
shows exact `iter_00020`, forbids `iter_00021`, and names the staged new H10
Crustle specialist. The historical Archaludon-to-Slowking source remains audit
evidence but cannot populate this active refresh card. This is display-only:
the selector, managed trainer, fleet, checkpoint, and handoff services are
unchanged. Receipt: `state/marnie_dashboard_handoff_fallback_r159.json`
(`sha256:81e2a62e7db522e0d423ca55a1f5172f628be9690da1d3bebb3310cc6d876d28`).

Revision 160 changes only the still-dormant post-Marnie Crustle bootstrap.
Train epochs 1--10 with Crustle's checksum-bound strategic guide under its
existing bounded ramp/hold schedule.  Then set guide loss, gradients, target
generation, fusion/action influence, serving authority, and gate authority to
exactly zero/off and run an exact 25-epoch expert refresh as epochs 11--35.
Only epochs 11--35 are final-checkpoint-selection eligible.  The exact frozen
Marnie iteration-20 H10 checkpoint must be checksum-bound as Crustle's
predecessor and included in Crustle's expert/practice opponent contract; it may
not be relabelled as a Crustle acting-seat action target.  Crustle bootstrap,
registration, or RL launch fails closed unless the 35-epoch receipt proves the
10+25 schedule and the completion-bound Grimmsnarl identity.  Active Marnie,
its fleet, guide retirement, iterations 19--20, and the no-iteration-21
boundary remain unchanged.

Revision 161 adds checksum-pinned expert-pilot importance to the complete
Crustle bootstrap, not only its guide-free refresh.  Every training epoch
1--35 samples the exact Crustle acting-seat expert corpus with the same bounded
top-100, same-training-split support tiers used for Marnie's expert bootstrap;
validation remains unweighted, unmatched or unverifiable pilots remain at
`1.0x`, and actions and causal labels are unchanged.  The guide remains active
only for epochs 1--10 and exactly off for epochs 11--35.  The completed Marnie
H10 checkpoint remains a weighted practice/holdout opponent anchor and is
never relabelled into Crustle expert actions.

The post-upload bootstrap now also fails closed unless Accepted Policy
Generation 15's action-conditioned latent lookahead remains enabled and
action-authoritative with its exact 512-wide, 0.25-capped, 412,130-parameter
inventory in the uploaded iteration-9 parent, every bootstrap epoch, and the
selected epoch-25 checkpoint. The frozen provenance and ready receipt carry
the same inventory. This runtime-inert preflight is bound by
The original runtime-inert preflight was bound by
`state/marnie_postupload_latent_policy_continuity_r137.json`; revision 140
supersedes it for the active bootstrap with
`state/marnie_postupload_latent_policy_continuity_r140.json`, which binds the
current bootstrap code, exact uploaded parent, owner-ceiling activation,
atomic migration, and permanent Marnie guide retirement. Neither receipt
alters the scheduler, fleet, or tail rule.

Revision 120 stages Marnie archetype-family generalization only after all
previously requested Generation-15 and fleet-tail changes.  The implementation
may be built and validated while training continues, but the family sampler and
its typed loss vector have no runtime authority until an immutable successful
iteration-9 Kaggle upload receipt is checksum-bound to the exact iteration-9
commit, checkpoint, singular package deck, bundle, competition, label, uploaded
file, and consumed one-shot authorization.  They activate atomically at the
first later clean boundary before collection starts, or chase the following
boundary without interrupting a started collection.  Missing observed legal
family lists, fewer than twelve non-package similarity clusters, missing study
evidence, a failed or inconclusive study, or an invalid upload receipt all keep
the current exact-list recipe active.  Kaggle games and scores remain strictly
evaluation-only, and the existing exact 60-card Marnie deck remains the sole
package and measurement deck.

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
- Active delegated Alakazam rule-derivative gateway:
  `goals/alakazam-elmo-rule-derivative/GOAL.md`
- Sole typed Alakazam rule-derivative authority:
  `goals/alakazam-elmo-rule-derivative/contract.json`
- Receipt-backed Alakazam rule-derivative execution-status projection:
  `goals/alakazam-elmo-rule-derivative/STATUS.json`
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
| 68 | 2026-07-30 | Stop extending the Archaludon ex public corpus and start training from the already stored, checksum-backed schema-7 days once that stored snapshot proves at least 16,639 exact games and all existing causal, guide, revision-56 route, bootstrap, terminal-preflight, and 128-worker launch receipts. Do not wait for unmaterialized later dates. In-progress partial day files are not training evidence. | Authorized for immediate managed activation. Stop the identity and guide expansion jobs only through systemd, seal a new immutable stored-day snapshot from completed daily receipts, build and validate the matching guide corpus for exactly those sealed dates, import it to Blackwell, then bootstrap/register/launch Archaludon at exactly 128 local workers and 128 games in flight. |
| 69 | 2026-07-30 | Supersede revision 68's corpus gate for the current Archaludon launch only: skip the unfinished schema-7 corpus seal for now and start immediately from the already imported protected Archaludon corpus. Preserve the unfinished schema-7 materialization as deferred, non-active evidence and do not imply that the older protected corpus satisfies revision 68's 16,639-game floor. Keep revision 56's 18 independently routed learned decision sources, strategic guide curriculum at `0.05`, and the exact 128-worker Blackwell profile. | Activated by the managed Archaludon revision-56 run `pure_rl_archaludon-ex_temporal1_8k_v2_20260730`. The live selector names `archaludon-ex`; the runtime registration receipt binds protected corpus `sha256:972a09b0be08c9de854724c4dc425921a1fb1ec00bb28a9b22cbefbe975ca0c2`, 18-source fusion-v2 checkpoint `sha256:57a3172d46e429cb9ba39187cf126f763fdbb10c15012ccdb31ef993010227be`, guide weight `0.05`, and the managed service launched at exactly 128 local workers. Schema-7 completion remains deferred and may not replace this run mid-iteration. |
| 70 | 2026-07-30 | Supersede revisions 65–66 for Blackwell only: reduce Blackwell from 128 to exactly `96` local simulator workers and `96` local games in flight. Set its local worker floor, target, ceiling, and live-pool maximum to 96. Preserve four environments per local worker and leave Elmo and every other remote worker allocation unchanged. This is a high-priority scheduler/memory-pressure repair. | Authorized for immediate managed activation during the uncommitted Archaludon iteration-0 collection. Update the canonical selector and restart only the managed trainer service; do not signal processes. Preserve the Archaludon checkpoint, corpus, fusion, guide, remote-worker, gate, and run identities. Record the new managed PID and memory-pressure recovery evidence after restart. |
| 71 | 2026-07-30 | Repair remote starvation immediately while retaining Blackwell at exactly 96 workers. Use the existing mid-iteration shared scheduler so completed remote work can claim from the common backlog instead of exhausting a fixed partition. Keep endpoint-owned Elmo/Bert reserves, but open one initial request socket per execution worker and permit at most one additional low-water wave (`POKEBOT_REMOTE_SOCKET_PREFETCH=1`, maximum `2`); the prior 5×/8× socket fan-out created hundreds of controller sockets without keeping server execution slots fed. Dashboard queue telemetry must be scoped to the current dispatch generation and may not reuse a prior generation's socket-prefetch values. | Activated through managed trainer restarts while Archaludon iteration 0 remained uncommitted. Live selector: Blackwell 96, mid-iteration scheduler enabled, Elmo/Bert 36/16 execution workers, socket fan-out 1×/2×. Current managed PID `502665`, `NRestarts=0`; cgroup pressure is zero. The dashboard parser now reports the live generation's 36/16 socket capacities and 336/176 controller reserves instead of stale 180/80 capacities and 192/112 reserves. |
| 72 | 2026-07-30 | Disable Elmo's routine whole-service rotation after 768 completed games. Keep 256-game child recycling and the independent RSS, host-free-RAM, worker-capacity, checkpoint-identity, and supervisor guards. A whole-service rotation drops every live trainer socket and produces a preventable CLOSE-WAIT/reconnect trough; it is reserved for an explicit managed drain or a real guard failure. | Activated first without restarting the Archaludon trainer. Elmo is healthy with 36 workers, routine rotation `0`, the fusion-v2-compatible durable image `poke-bot-truenas-worker:archaludon-fusion-v2-r72`, and the original 30 GiB RSS/24 GiB host-free-RAM guards. When the already-damaged pre-repair iteration-0 dispatch later stopped making material progress at 6,571/7,168 with full memory pressure near 51% and Elmo idle, the uncommitted dispatch was cleared through the managed trainer service. The immutable seed and all runtime identities were preserved, stop protection was restored, memory pressure returned to zero, and fresh Elmo/Bert work was observed under PID `604944`. Receipts: `state/elmo_remote_lifecycle_repair_r72.json`, `sha256:5f492f245a4f4b0b3b5686980080bddcdfcb84ac3fa189a1cd5dea9a23896480`; `state/archaludon_dispatch_recovery_r72.json`, `sha256:a809d80d270c695a34c5504ceac938664b1abb62a21dc702f5a54f2e21d00ed8`. |
| 73 | 2026-07-30 | Keep Blackwell hard-pinned at exactly 96 simulator workers, but repair its repeated cgroup pressure by reducing the independently tunable local inference-leaf topology from 10/24 GPU0/GPU1 replicas to 4/12. The repeated 110 GiB condition is dominated by 34 spawned CUDA leaf processes plus 96 simulator processes, not by the replay/result buffer; leaf replicas are not simulator workers. Preserve both local GPUs, four environments per simulator, all remote allocations, the Archaludon seed/checkpoint/corpus/fusion/guide/run identity, and the 96/96 worker invariant. Also classify an active collection with zero unassigned games and outstanding claimed results as draining, never remote starvation; ping-cache age remains unrelated. | Activated through the managed trainer service while iteration 0 remained uncommitted. PID `691996` runs 96 simulators and exactly 4/12 GPU leaves; both GPUs reached 99%/86% during collection, local throughput rose to 7.02 GPS, cgroup memory fell from 110.0 GB to 62.2 GB, host available memory rose from 4.1 GB to 42.3 GB, and memory pressure/high/OOM counters are zero. Stop protection is restored. Elmo was recreated from `poke-bot-truenas-worker:archaludon-fusion-v2-r72` and is healthy with the same checkpoint and 36 workers. Receipt: `state/archaludon_memory_topology_repair_r73.json`, `sha256:16ba9c92d6c4fd5f2c05b1e8f2ee696a7334fc3d46d122b69bb77fada2208ad6`. |
| 74 | 2026-07-30 | Correct Elmo's memory guard without increasing its allowance or weakening a safety boundary. The 36-worker/4-leaf endpoint twice exited when summed descendant RSS reached `30.00` GiB, but the container's exact cgroup-v2 peak was only `24.70` GiB: spawned processes share model and library pages, so adding every process RSS double-counts those pages and falsely classifies healthy memory as a leak. Keep summed process-tree RSS as diagnostic telemetry only. Apply the existing `30` GiB inner guard to cgroup-v2 `memory.current`, fall back to process-tree PSS when cgroup-v2 is unavailable, and use summed RSS only as the final fail-closed fallback. Preserve the independent `24` GiB host-available floor, `64` GiB/no-swap container cap, 256-game child recycling, worker-capacity/checkpoint/supervisor guards, 36 Elmo workers, and disabled routine whole-service rotation. | Activated at a verified zero-active-job boundary during Archaludon iteration-3 training through the managed container lifecycle. A first full-workspace image was rejected before serving because unrelated current router bytes failed the pinned public-tree contract; exact r72 was restored, then the valid candidate was rebuilt as a one-file overlay whose complete r72 layer chain, `poke_bot` tree, entrypoint, supervisor, checkpoint, 36-worker/4-leaf identity, and trainer PID/restart count are unchanged. Post-activation continuity is verified across the complete 4,000-game iteration-3 formal heldout dispatch: 2,368 local plus 1,632 remote games drained at 10.31 GPS, Elmo remained healthy with zero restarts, zero failed jobs, zero cgroup memory events, and a 21.49 GiB cgroup charge below the unchanged 30 GiB inner guard while summed RSS remained diagnostic-only. Receipts: `state/elmo_memory_guard_repair_r74.json`, `sha256:50ab7e11f9c6f8cf81650e90a197c5027dde4fc3b54e906c96286a4acef46e84`; `state/elmo_memory_guard_collection_continuity_r74.json`, `sha256:5f05e84150a014be3558e302421dd6cfb5715bd9b78d301abf1aa3e2265c621a`. |
| 75 | 2026-07-30 | Release Blackwell's device-resident training CUDA cache before rebuilding inference leaves or beginning promotion/formal/diagnostic evaluation. The long-lived trainer retained about `42.19` GiB on GPU 1 after iteration-3 training, leaving the 12 evaluation leaves unable to allocate 32 MiB and causing the research-control exact-row audit to reject the run at 814/1,000 valid games. After the replay dataset is deleted and the host heap is collected, synchronize the training CUDA device, record reserved bytes before/after, call `empty_cache`, then rebuild the unchanged 4/12 leaf farm. Keep Blackwell at exactly 96 simulator workers, preserve the candidate/checkpoint/corpus/guide/fusion identities, and never convert failed leaf games into evaluation evidence. | Activated through the managed trainer service at the immutable iteration-3 collection/candidate recovery boundary with Elmo and Bert at zero active jobs. The invalid diagnostic evaluation remained uncommitted. PID `1387831` loaded the patched source, stop protection is restored, and the exact preserved candidate completed 4,000/4,000 formal games plus 1,000/1,000 research-control games with both audits passing, no CUDA OOM or fail-closed leaf events, 68.7% formal win rate, and 57.8% diagnostic win rate. Iteration 3 committed and iteration 4 started without recollection or retraining. The first ordinary nonzero sample is verified: iteration 4 released CUDA-reserved memory from 45,116,030,976 bytes to 104,857,600 bytes before rebuilding the unchanged 4/12 leaf farm. Its complete post-release evaluation then committed with 4,000/4,000 exact formal rows, a passing 68.51% skill-weighted active gate, and 1,000/1,000 disjoint diagnostic rows at 59.9%. Because iteration 4 was below the terminal floor, the managed loop retained the pass and correctly began iteration 5 without a CUDA OOM, failed-leaf event, traceback, service restart, or runtime identity change. Receipts: `state/blackwell_cuda_cache_boundary_repair_r75.json`, `sha256:880bde6079dbe7471d21d77ec164f5c5c0cc1757cc36fd23dd41634fbf3aadcb`; `state/blackwell_cuda_cache_boundary_continuity_r75.json`, `sha256:b517a04bf544866445fa16220380a2b383c1e83f2245e1791318280699ed49d6`; `state/blackwell_cuda_cache_evaluation_continuity_r75.json`, `sha256:fa2da0f04d15a820b8d137c0edc0bd91e9851b983c41364c542386ae4b247a8a`. |
| 76 | 2026-07-30 | Preserve revision 69's already-imported protected Archaludon corpus across the iteration-5 expert-rehearsal boundary. Its feature shards use dataset schema 6; active schema 7 added only exact setup `SelectContext` and demonstrated STOP metadata. Permit schema-6 protected expert shards through an explicit fail-closed compatibility path that keeps every pre-existing causal target unchanged and maps only the unavailable setup context to UNKNOWN so the new setup objective is masked. Never fabricate setup labels, relabel a CPU pack, change the runtime split, recollect the completed iteration, or turn the guide into imitation. | Activated after iteration 5 had immutably committed 8,192 games and 417,582 decisions but before its learner update. The exact runtime-keyed packing-schema-5 CPU pack was rebuilt from stored protected data with 83,980 decisions, then loaded as a checksum-verified cache hit. The five-epoch rehearsal committed checkpoint `sha256:41ea7c7de8f7fabaf48b25b6dce11f9d47d6d128f68d21f95f22905902a9dafd`; both 8,192-sequence replay shards loaded, and iteration-5 training resumed over 16,384 sequences with 17,188 guide rows at weight `0.05`. The collection receipt remains byte-identical, no games were recollected, and Slowking remains ready at the automatic post-Archaludon boundary. Receipt: `state/archaludon_expert_rehearsal_schema6_compat_repair_r76.json`, `sha256:4d9b6c22d93551f716048f914cac45ffafca7a14dec889626756b46fc187e837`. |
| 77 | 2026-07-30 | Repair Slowking's circular combo gate without weakening revision 58. The immutable implementation, exact causal corpus, resident CPU pack, and parameter-inventory receipts authorize creation of one checksum-bound supervised candidate only; they never authorize freeze, registration, selector activation, or RL launch. Train the exact 25-epoch candidate first, then generate `poke_bot.slowking_combo_head_validation/v1` from that candidate and held-out Slowking states before freeze. The final receipt must bind the exact candidate checksum and prove typed loss/calibration, finite nonzero combo-head and route gradients and updates, causal/option-conditioned bounded influence, leave-one-head-out logit/action attribution, exact parameter inventory, memory/latency, loadable package compatibility, and used capacity. Any failed or missing final check rejects the candidate and leaves production unchanged. | Activated in the handoff implementation and canonical protocol after 66 focused tests passed. The managed handoff remains stopped while its generated contract is rebound to the new two-stage authority split; no unvalidated Slowking checkpoint has been frozen, registered, selected, or launched. |
| 78 | 2026-07-31 | Activate the candidate-bound validated Slowking runtime with every registered learned head, including `setup_board_outcome` and `combo_state`, present in the exact fused action path. Treat specialist runtime-registry `version` as a positive monotonic content revision under the unchanged `/v1` schema, require the canonical learned-head prefix without permitting omission or reordering, and bind additional specialist heads exactly through the registry, checkpoint inventory, and selector architecture flags. Preserve failed uncommitted seeds and remote rotation history rather than treating either as training evidence. | Activated at Slowking iteration 0. Selector `slowking` uses the plan-v30 runtime, the validated 1,910,963-parameter Model Format 6 checkpoint, 19 registered fused heads, exactly 96 Blackwell simulator workers with 4/12 leaves, and 36/16 Elmo/Bert workers. Both remote workers serve the 20-route Slowking tree; the between-iteration hard gate and formal-remote verification passed before collection began. Receipt: `state/slowking_runtime_activation_repair_r78.json`. |
| 79 | 2026-07-31 | Consider Slowking a failed experiment and move directly to the Alakazam phase. Stop the managed Slowking trainer and pass-gate handler immediately. Preserve iteration 5's failed formal gate, iteration 6's sealed 8,192-game collection, and all uncommitted iteration-6 work, but do not publish, freeze, register, submit, or serve a Slowking checkpoint. Replace the former 15-frozen G0 premise with a truthful terminal-fleet disposition: 14 required specialists frozen plus one explicit `failed_experiment` Slowking exception. This owner exception authorizes final-format Alakazam preparation but does not relabel Slowking as passed or grant it completion credit. | Activated immediately at the stopped Slowking iteration-6 boundary. The managed trainer is stopped with restart protection restored, the pass-gate handler is inactive, no iteration-6 checkpoint or commit exists, and Slowking is absent from the frozen registry. Activation receipt: `state/slowking_failed_experiment_transition_r79.json`. Alakazam G0/G1 issuance and launch follow this receipt-backed boundary. |
| 80 | 2026-07-31 | Make final-format Alakazam a high-volume final-submit run rather than an ordinary 15-iteration specialist. Collect exactly 32,768 games per iteration with revision 79's deterministic 50/50 seat split, retain one learner epoch per iteration, and permit up to 189 iterations (6,193,152 training games) so training continues until strong evidence exists rather than accepting an iteration ceiling. Freeze, register, and submit only after (a) a complete audited 4,250-game premium gate reaches at least `0.65` skill-weighted win rate and a `0.60` lower confidence bound, (b) the disjoint official controls reach at least `0.60`, and (c) a separate rating calculation over actual balanced-seat simulations against checksum-bound frozen agents with known Kaggle ratings has a 90% lower bound of at least `1000`. The 65% strength gate and 1,000-rating simulation are independent checks. | Staged before the first H10 RL game. Materialize an immutable Alakazam-only gate derivative, bind both checks into the isolated runtime registry and committed gate result, validate the same terminal handler, rotate idle remotes to the H10 checksum, and activate through the managed Alakazam service. The ordinary LC50 fallback and ceiling acceptance are forbidden for this refresh. |
| 81 | 2026-07-31 | Supersede revision 80's final-format Alakazam iteration size only: collect exactly 16,384 games per iteration, the nearest power of two to the requested approximately 16,000. Preserve one learner epoch per iteration, the deterministic exact 50/50 assigned/actual/consumed seat split, the maximum 189 iterations, all three independent terminal gates, the 96-worker Blackwell profile, and Matchup Router Format 6. With the unchanged iteration limit, the maximum training-game budget becomes 3,096,576. | Activated through the isolated managed H10 service at the uncommitted iteration-0 boundary. Runtime registry version 6 binds 16,384 games, one epoch, 189 iterations, the unchanged three terminal gates, and no production-selector authority. Receipt: `state/final_format_alakazam_h10_iteration_size_r81.json`; activated at `2026-07-31T17:00:08Z`. |
| 82 | 2026-07-31 | Set final-format Alakazam's training mix to exactly one-eighth self-play and seven-eighths specialist/public-opponent play. For each 16,384-game iteration this is exactly 2,048 self-play games and 14,336 public-opponent games. Repair the exact-seat scheduler so this mix still produces exactly 8,192 first-seat and 8,192 second-seat assigned, retained, and consumed games. Preserve all training/evaluation isolation: the 4,250 premium gate, 1,000 official controls, and rating simulations remain additive evaluation and never enter training. | Activated under managed PID `262970`, restart count 0. The live collection plan is 2,048 self-play, 8,168 strong-public practice, and 6,168 diverse-public games; exact global seats and per-opponent balance passed before collection. Bert remains CPU after an exact H10 Apple MPS comparison. Receipt: `state/final_format_alakazam_h10_mix_r82.json`; activated at `2026-07-31T17:00:08Z`. |
| 83 | 2026-07-31 | Use eligible remote simulator capacity throughout final-format Alakazam collection and evaluation rather than reserving it for self-play. Elmo participates in self-play, public mix, holdout, and evaluation immediately. Bert is temporarily removed only for the owner-requested 512-game exact-H10 Apple CPU/MPS benchmark, then rejoins the same phases with its checksum-identical worker. Preserve the 96-worker Blackwell floor, exact seat schedule, training/evaluation isolation, and all checkpoint identities. | Staged during the uncommitted iteration-0 public-mix collection. The active process inherited the historical local-only default, so activation requires the next managed start after Bert's benchmark and reattachment; no committed iteration or checkpoint may be discarded. |
| 84 | 2026-07-31 | Raise Elmo's production process-tree memory guard from 30 GiB to exactly 45 GiB. Preserve the independent 24 GiB minimum host-free-RAM guard, 64 GiB no-swap container ceiling, 36-worker/4-leaf topology, exact active H10 checkpoint, and all fail-closed worker-health and capacity checks. | Activated immediately after the 30.01 GiB guard tripped while the host still had about 55 GiB available. The managed container now enforces 45 GiB, reloaded exact checkpoint `sha256:e65123a13abb61332fe89e66946103a83c766e2f15315b945bfe9b6bf0c2d32e`, passed health with 36 workers and four leaves, and the managed trainer resumed iteration-0 self-play at 96 local plus 36 Elmo workers. |
| 85 | 2026-07-31 | Reduce final-format Alakazam's learner decision cap from 8,192 to exactly 6,144 decisions for both warmup and ordinary RL batches after GPU 1's CUDA expandable-segment allocator repeatedly failed to map an additional 20 MiB and the first attempt ended with an illegal-memory-access fault. Keep the independent 240-game cap, one learner epoch, exact 16,384-game corpus, exact seat split, all learned heads, matchup-adapter epoch, and terminal gates unchanged. Reuse the already completed checksum-bound iteration-0 collection and replay cache; do not recollect public mix or self-play. | Activated through the managed Alakazam service at `2026-07-31T18:44:30Z`. PID `588148`, restart count 0, launched from runtime registry `specialist_runtime_registry_h10_r85_batch6144_all_remotes.json` (`sha256:9e967683679bdbeeb375955c88fc0369a59e0ef29b2e2fb07183864ff636a24e`) and resumed the completed iteration-0 collection without recollection. The 6,144 cap is active but remains under observation because the allocator still emitted nonfatal mapping failures during ep0. |
| 86 | 2026-07-31 | Supersede revision 83's Bert restoration condition: keep Bert completely out of production until a new 512-game exact-H10 MPS benchmark finishes using the optimized inference implementation, including four MPS leaves, 16 workers, per-worker home-leaf affinity, MPS autocast, disabled per-batch MPS cache eviction, and the 45 GiB benchmark guard. A prior 512-game MPS result from the older staged inference bytes is not sufficient. Keep Elmo and Blackwell production running independently. After the optimized receipt completes, select CPU or MPS from measured whole-game throughput and runtime validity before restoring Bert. | Activated for isolation at `2026-07-31T18:56:01Z`: managed LaunchAgent `com.pokebot.remote-worker-8766-h10-r80` is unloaded and port 8766 is closed. The optimized BF16 canary completed 4/4 games without errors; the full isolated 512-game run is active on loopback port 8776 with checkpoint `sha256:e65123a13abb61332fe89e66946103a83c766e2f15315b945bfe9b6bf0c2d32e`. Production restoration remains blocked pending the completed benchmark receipt. |
| 87 | 2026-07-31 | Optimize future `rl-prep baseline` passes without increasing the learner decision cap. First use a value-only frozen-baseline forward path that bypasses option decoding, decision fusion, guides, auxiliary/strategic heads, log-softmax, and AWR diagnostics. Add a bounded one-batch CPU prefetch. Generalize the existing padded, length-bucketed causal temporal mechanism as a separately gated second optimization; it may activate only after an exact-device comparison proves identical cached values and subsequent AWR weights against the reference path. Do not wrap the existing CUDA model in competing Python workers. Multi-GPU baseline-only sharding is optional later work after these paths validate. | Staged for the next safe learner boundary; the active ep0 process is unchanged. The exact-temporal value-only implementation and bounded prefetch pass 10 focused AWR tests, including bit-identical cached values and derived weights on the CPU reference. Multi-game packed temporal inference is implemented but remains disabled by default because the CPU shadow comparison exposed last-bit FP32 differences despite numerical closeness; it requires an exact Blackwell parity receipt before activation. |
| 88 | 2026-07-31 | Recover final-format Alakazam from the iteration-0 adapter-phase CUDA OOM without recollection. The ordinary 6,144-decision epoch completed, but the first dormant-matchup-adapter batch attempted a 3.80 GiB allocation with only 3.32 GiB free. Supersede revision 85's learner cap with 4,096 decisions for warmup and ordinary batches, give adapter-only training an independent 2,048-decision cap, release CUDA cache before adapter fitting, and recursively split only an OOMing adapter batch. Preserve the exact completed 16,384-game corpus, seat split, all learned heads, one adapter epoch, and terminal gates. | Staged at the failed uncommitted iteration-0 boundary. The managed auto-restart loop is stopped with MainPID 0; activation requires a checksum-bound runtime registry and a successful managed resume that reuses the completed collection and advances adapter training beyond its first batch. |
| 89 | 2026-07-31 | Keep Bert out of final-format Alakazam through the remainder of iteration 0. The learner and dormant-adapter recovery are Blackwell-local and must not wait for Bert. Use Elmo as the sole configured remote for the iteration-0 recovery runtime. After iteration 0 commits, restore Bert to all eligible simulation phases only if its completed optimized exact-H10 MPS receipt remains valid and the managed worker is stable. | Activated through the managed Elmo-only recovery registry `specialist_runtime_registry_h10_r89_iter0_elmo_only_batch4096_adapter2048.json` (`sha256:661450f4e08f6e1f8bea5184ad1fea0f4bb3b85d34f8d04550591277918e1bf2`). PID `706275`, restart count 0, passed the hard gate with exactly one remote, reused the completed iteration-0 collection, and entered the 4,096-decision learner epoch. LaunchAgent `com.pokebot.bert-post-alakazam-iter0-rejoin-r89` now polls only for the immutable iteration-0 commit, then requires two stable optimized-MPS health probes before performing the managed post-commit registry/service migration. |
| 90 | 2026-07-31 | Reduce final-format Alakazam's ordinary and warmup learner decision cap from 4,096 to exactly 2,048 after dense ordinary batches repeatedly exhausted GPU 1 and poisoned the CUDA context with an illegal-memory-access failure. Keep the independent 240-game cap and the dormant-matchup-adapter cap at 2,048, preserve the completed 16,384-game iteration-0 corpus, exact seat split, one learner epoch, all heads, guide weight, gates, Elmo-only iteration-0 remote policy, and post-commit Bert admission rule. Do not recollect. | Activated through managed PID `796696`, restart count 0, using Elmo-only registry `specialist_runtime_registry_h10_r90_iter0_elmo_only_batch2048_adapter2048.json` (`sha256:0b392e964d17508fe2cfc76bda98f126a8947b466a83114bf868a4eb7df69c40`). The live trainer command contains ordinary, warmup, and adapter caps of exactly 2,048 and is reusing the completed iteration-0 replay cache. The post-commit Bert registry was also rebound to the 2,048 caps (`sha256:8e45fbdf2f97ea8c978127b72aca352df39752c41715cd2e280b667e0b9549a5`). |
| 91 | 2026-07-31 | Speed agreement evaluation independently from optimizer memory safety. Parent/candidate agreement retains only deterministic policy argmaxes, so use a policy-only forward that bypasses value, guide, AWR, auxiliary-loss, and diagnostic calculations, and use the previously completed 6,144-decision inference cap while ordinary/warmup optimization and adapter fitting remain at 2,048. Also select Bert's post-iteration-0 backend from exact-H10 whole-game throughput: 16 simulators, four CPU leaves, four threads per leaf. Eight threads and optimized MPS remain valid measured alternatives but are slower in completed games per second. | Agreement parity passed on the focused CPU test and is staged on Blackwell for the next process boundary without interrupting the current iteration. Bert's repaired staged worker completed all topology probes without errors: CPU-1t 0.353 GPS, CPU-2t 0.374 GPS, CPU-4t 0.596 GPS, CPU-8t 0.572 GPS; optimized 512-game MPS was 0.304 GPS. The post-commit registry binds CPU-4t plus the policy-only agreement contract and is subsequently superseded only for revision 92's future optimizer cap. |
| 92 | 2026-07-31 | After iteration 0 completes at the recovery-safe 2,048 optimizer cap, use exactly 3,072 decisions for future ordinary and warmup learner batches. The observed 2,048 batch used about 29.4 GiB of GPU 1 and left about 19 GiB free, while 4,096 repeatedly exhausted the device and poisoned its CUDA context. Preserve the independent 2,048 adapter cap and the policy-only 6,144 agreement cap. Do not restart or alter the healthy active iteration 0. | Staged in the post-iteration-0 configuration, then superseded only for the adapter cap by revision 93 before activation. |
| 93 | 2026-07-31 | Use 3,072 decisions for future dormant-matchup-adapter batches as well as future ordinary/warmup learner batches. The repaired iteration-0 adapter phase at 2,048 used about 20.6 GiB and left about 27.8 GiB free after its explicit cache release, so the measured middle cap has sufficient headroom. Preserve recursive adapter OOM splitting and the policy-only 6,144 agreement cap. Do not alter the active iteration-0 adapter epoch. | Staged in post-iteration-0 registry `sha256:9e58b74b1f0ffb61bbe4ed704a86d4dc45dde3fee4b74eb86e43653fa49bca54` and canonical configuration while the current 2,048 adapter epoch continued without restart. |
| 94 | 2026-07-31 | Raise final-format Alakazam's terminal premium skill-weighted win-rate requirement from `0.65` to `0.75`. Preserve the separate `0.60` skill-weighted confidence lower bound, `0.60` official-control non-regression floor, and independent simulated Kaggle-rating 90% lower bound of `1000`. Do not reinterpret or terminate the already-running iteration-0 evaluation under its immutable 65% gate contract. | Staged for the first safe post-iteration-0 process boundary through a new immutable 75% gate derivative. The active gate file and healthy managed iteration remain unchanged. |
| 95 | 2026-07-31 | Build and queue exactly one first-preferring Kaggle submission from final-format Alakazam's exact committed zero-indexed iteration-4 checkpoint. Treat it as an explicitly authorized training-milestone snapshot, not a freeze, terminal gate pass, production selector change, or completion event. Packaging and queue processing are asynchronous and may never stop, pause, or block continued iteration-5+ training; quota and spacing waits remain pending under the second-most-recent logical-submission policy. | Staged as an idempotent managed watcher that activates only after the immutable iteration-4 commit exists and checksum-binds the checkpoint, exact 60-card Alakazam deck, matchup tree, bundle, and one-shot queue row. |
| 96 | 2026-07-31 | After the iteration-4 milestone submission, also build and queue one first-preferring final-format Alakazam snapshot after every tenth zero-indexed training iteration (`10`, `20`, `30`, …, `180`) that durably commits before terminal completion. Each snapshot is independently checksum-bound and one-shot authorized. No milestone submission freezes a checkpoint, changes the selector, claims a gate pass, or stops/pauses/blocks training. | Supersedes revision 95 only by making the managed milestone watcher recurring. Iteration 4 remains required; later eligible commits are discovered idempotently and queue asynchronously under the unchanged quota/spacing policy. |
| 97 | 2026-07-31 | Supersede revision 96's ten-iteration snapshot cadence with one snapshot at the commit immediately before every fifth subsequent zero-indexed iteration: `4`, `9`, `14`, `19`, …, `184`. This means the first bundle is created from committed iteration 4 before iteration 5's checkpoint can exist. Preserve one first-preferring copy per milestone, exact checksum binding, asynchronous queueing, and the rule that submission work never freezes, stops, pauses, or blocks training. | Staged in the recurring managed watcher before any eligible commit exists. The watcher discovers eligible durable commits idempotently and leaves spacing/quota waits to the nonblocking queue processor. |
| 98 | 2026-07-31 | Submit one additional first-preferring final-format Alakazam iteration-0 snapshot to Kaggle, then continue iteration 1 from the promoted iteration-0 learner without pausing training. Preserve the ordinary revision-97 cadence beginning at iteration 4. | Activated at the safe iteration-0 boundary. Iteration 0 promoted checkpoint `sha256:8f8ab3adee66416a7400f73e8fb0fe2749b183976407c8c7f5d28cb552878981` after 322 matchup-adapter steps over 718,133 rows and 15 populated routes. Its checksum-bound bundle `sha256:8896797abbd21585ed5d4db4e7621a73e6add792db837f59ab4701c8d4cc603b` passed isolated smoke tests and was submitted as Kaggle submission `55146726`. Iteration 1 resumed from that checkpoint with one adapter epoch enabled, both remotes verified, and the revision-94 75% gate active. The superseded pre-Bert retry plan was preserved in quarantine. |
| 99 | 2026-07-31 | Final-format Alakazam must complete at least 11 training iterations before terminal eligibility. Raise the terminal handler floor from completed iteration 5 to completed iteration 11; preserve the 188 ceiling, 16,384-game exact 50/50 seat contract, 75% strength gate, 60% confidence and official floors, separate 1000 rating lower bound, and nonblocking milestone cadence. | Activated immediately in the checksum-bound terminal-handler registry and in the next-start service definition. The healthy active learner is not restarted; its current iteration completes under its immutable launch arguments, while the handler cannot authorize an early terminal transition. |
| 100 | 2026-07-31 | Supersede the prior final-format Alakazam simulated Kaggle-rating lower bound of 1000 with 1150. Preserve the independent 75% premium strength requirement, 60% confidence and official-control floors, exact 50/50 training seats, completed-iteration-11 terminal floor, and all nonblocking milestones. The owner’s transient 1300 request was corrected to 1150 before activation and creates no runtime artifact. | Activated immediately through a new immutable 1150-rating gate and terminal-handler registry; the healthy active learner remains uninterrupted. |
| 101 | 2026-07-31 | Clarify final-format Alakazam’s exact 50/50 seat contract: it governs assigned and retained source games, not the derived replay-sequence projection. A self-play source game may correctly emit both player perspectives and a replay window may span adjacent immutable shards, so replay sequence counts are integrity-audited for valid seats and bound identity but are not required to be numerically even. Preserve the exact 8,192/8,192 source-game schedule, all collection receipts, and terminal gates. | Activated as a receipt-semantics repair at the uncommitted iteration-1 boundary. The immutable collection and replay cache are reused; no games are recollected or discarded. |
| 102 | 2026-08-01 | Recover the final-format Alakazam learner from recurring Blackwell allocator exhaustion. Set the ordinary and warmup learner decision cap to exactly `2,048`; retain the independent dormant matchup-adapter cap at `3,072`, the policy-only agreement cap at `6,144`, the 240-game cap, all 19 fused learned heads, the exact 16,384-game corpus, and the hard 96-worker Blackwell profile. A sealed collection and completed expert rehearsal must be reused, not regenerated, during this recovery. | Activated at the uncommitted iteration-5 receipt-backed recovery boundary by immutable runtime registry `specialist_runtime_registry_h10_r102_learner2048_adapter3072_rating1150_minimum_iter11_all_remotes.json` (`sha256:8e3429b08e65cb3e48ff4389e9f1fe674b72bc22e8c7fd39b963d4fec86216a3`). Iteration 5 retained its sealed 16,384-game collection and exact 8,192/8,192 source-seat receipt; the completed five-epoch rehearsal `sha256:80a5bd4da2350b3aecff8b8fb85917990408a30c4655dbd76f97032418c83719` is reused after receipt validation. |
| 103 | 2026-08-01 | Finish final-format Alakazam through the durable zero-indexed `iter_00020` commit, freeze and register that exact checkpoint, collect no `iter_00021`, and then begin a new separately versioned, up-to-date Marnie's Grimmsnarl ex refresh. If iteration 20 passes every existing strength, confidence, official-control, and rating gate, retain the measured-pass disposition; otherwise use explicit owner ceiling acceptance, preserve every failed gate result, and never relabel it as a measured pass. | Boundary activation is staged while iteration 13 collects; the active learner was not interrupted. Immutable terminal registry `specialist_runtime_registry_h10_r103_iter20_terminal.json` is `sha256:f02d23b9e6a0966ebcc6099cb71e4129e9d497538ae8990c7ed8119135bf4021`; staging receipt `final-format-alakazam-r79-iter20-transition-stage-r103.json` is `sha256:202ce1e9824237eccc4b671d591c1038145f820a633f13e35a879f4aebfd4a7f`. The revised handler is live with floor/ceiling 20, and the managed boundary watcher is waiting for `commits/iter_00020.json` to checksum-match `loop_state.json` before stopping the trainer. The truthful pass-versus-ceiling finalizer and checksum-valid Grimmsnarl static handoff are installed; no iteration 21 is authorized. |
| 104 | 2026-08-01 | Starting at Alakazam's next clean committed boundary, use H10-I Fusion v3: every learned head contributes through a typed-output-centered option route with learned positive reliability bounded to `[0.25, 4.0]`; cap the currently unidentifiable `action_type` route at `0.25` until a versioned target repairs its labels. Replace direct guide imitation with `strategic_directional_v2`: guide preferences at weight `0.05` rank only the causal `action_q`, `action_resource`, `action_utility`, `setup_board_outcome`, and available `combo_state` routes, while final policy logits remain outcome/RL trained. The later Marnie's Grimmsnarl ex refresh must use the same H10-I `7/3/7`, FF-2496, 512-wide head-residual architecture, 19 learned heads/routes, Fusion v3, and directional-guide contract. | Activated for Alakazam iteration 14 by design-migration receipt `migration_0021.json`; the exact Fusion-v3 learner `sha256:49882f0a1ce255448de2559d32ced3d26b5de5ee5070044edcfab8f238234acd` passed the between-iteration hard gate on Blackwell, Elmo, and Bert. Marnie remains pre-staged without selector or gradient authority until Alakazam freezes/registers at iteration 20. Its r104 check receipt forbids pre-H10 training and requires H10-I `7/3/7`, FF-2496, residual-512, 19 learned heads/routes, Fusion v3, Router Format 6, matchup adapters, and the directional guide before managed RL may launch. A direct pre-training shape canary now proves that the accepted-core parent can migrate straight into the Marnie-labeled H10/Fusion-v3 child without a pre-v3 training phase: checkpoint `sha256:f6b80ebda1b6c12a3aaa748000a06f2bce17598596af80622ec933d4198448b3`; validation receipt `state/final-format-marnie-r104-h10-fusion-v3-shape-canary.json`, `sha256:bfc721d6656ff38db3a803ebbd888b8883974f7a9cc3c1a36f3b519aac65c34c`. The complete dormant managed chain is installed and validated by `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-managed-chain-stage-v13.json` (`sha256:dd61cc5c09a481f80c1877e899632c9c1c8051c6b819ee3672154777a5c4c15f`): it checksum-binds the Alakazam-completion handoff, normal post-Alakazam cumulative-core refresh attempt, direct H10/Fusion-v3 materialization, exact 25-epoch Marnie bootstrap, exact-checkpoint H10 registration, managed H10 RL launch, gate-handler launch, truthful refresh completion registration, the required `post_refresh_sequence_complete_for_capacity_v2` barrier, and the corrected 14-member current-plus-history population release and trainer. Before a single Marnie RL game can start, the RL unit preflights the same exact terminal freeze/package/submission/completion path that will run after its gate. The registration and RL preflights re-open the checkpoint and require H10-I `7/3/7`, FF-2496, residual-512, all 19 learned heads/routes, Fusion v3, Router Format 6, matchup adapters, and `strategic_directional_v2` at weight `0.05`; the RL unit independently pins the same architecture environment, exact 64-slot Router-6 registry path, and exact 96-worker Blackwell collection/heldout profile. All eleven services are loaded and inactive until the iter-20 completion receipts exist. Earlier v1-v12 receipts remain immutable partial staging evidence and do not authorize training. The canary and staged services have no current selector, runtime, gradient, or submission authority. |
| 105 | 2026-08-02 | Recover Alakazam iteration 15 after the 2,048-decision Fusion-v3 optimizer exhausted GPU 1, emitted allocator mapping failures with only 14–27 MiB free, and poisoned CUDA with an illegal-memory-access failure. Use exactly 1,536 decisions for ordinary and warmup optimizer batches while retaining the independent 3,072-decision matchup-adapter cap, 6,144-decision policy-only agreement cap, 240-game cap, all 19 fused heads/routes, exact 16,384-game corpus, and hard 96-worker Blackwell profile. Reuse the sealed iteration-15 collection, exact seat receipts, replay caches, and completed five-epoch rehearsal; recollection or rehearsal regeneration is forbidden. This Alakazam memory recovery does not weaken or alter the staged Marnie H10-I architecture. | Activated at the uncommitted iteration-15 receipt-backed boundary. Design migration `migration_0023.json` (`sha256:2c9d964c2416d16a99ee1f65d01d753c36881cd24efb34cd2651d1bfd0fa1f28`) preserved the sealed 16,384-game collection; the completed rehearsal and exact seat index `sha256:d11d22c4371d927c36c63f71590ed4a4997c66b06694ff96cef11bcc473dd90f` were reused without mutation. The managed service resumed with zero restarts, completed all learner and 964 adapter batches, sealed the exact iteration-15 checkpoint `sha256:aea647346e6a6a0819d983c56326fdd393ef385d662ba1f1e7110742116f7bb1`, and started iteration 16 from that checkpoint. Activation receipt: `state/final_format_alakazam_iteration15_allocator_recovery_activation_r105.json`, `sha256:ae4c9e349c2429de33e24fce87a875eed684e27b40cd38f6e68ba1d4f58bac59`; completion receipt: `state/final_format_alakazam_iteration15_allocator_recovery_completion_r105.json`, `sha256:6a60b657396f5917f759b321bc4e1f4e5c040ea1e1d2dd717458bc55b314fc41`. |
| 106 | 2026-08-02 | Make the exact 96-worker Blackwell profile include formal heldout dispatches, and repair the restart-time Fusion-v3 design contract without changing model behavior. Formal heldout local workers are exactly `96`. The iteration-17 learner's two typed-routing fields and 19 learned route-reliability scalars are existing checkpoint architecture, so the design ledger may repair only missing→`true`, missing→`0.25`, and parameter telemetry `10,645,166`→`10,645,185`; every wider change remains forbidden. Preserve Alakazam's iter-17 commit, all gates, exact iter-20 ceiling, and the staged Marnie H10-I architecture. | Activated at the exact iteration-17→18 receipt-backed boundary. Worker receipt `state/final-format-alakazam-r79-heldout96-boundary.json` is `sha256:c4fc1caacad1c3627c8d7c61710e57404c542394107207de9db6403145a9d3ce`; exact schema-repair design migration `migration_0025.json` is `sha256:66a45ee43cc52e040bc1264a2f6e342c022627f8906c996056410b36349aef0e`. Managed PID `3580748` resumed iteration 18 with zero restarts and exact `SIM_WORKERS=96`, `GAMES_IN_FLIGHT=96`, `HELDOUT_LOCAL_WORKERS=96`, worker floor/ceiling 96, and live-pool maximum 96. The interrupted uncommitted partial iteration-18 attempt remains quarantined. Marnie's dormant v13 chain independently pins H10-I `7/3/7`, FF-2496, residual-512, 19 heads/routes, Fusion v3, Router Format 6, and the same exact 96 collection/heldout profile. |
| 107 | 2026-08-02 | Give the final-format Marnie's Grimmsnarl ex refresh the same nonblocking training-snapshot submission structure, with the owner's clarified cadence: queue one exact `first_if_allowed` Kaggle copy from durable zero-indexed iteration `0`, then from commits `4`, `9`, `14`, and every later `5n+4` commit that exists before the refresh completes. Each snapshot is checksum-bound to its exact checkpoint, commit, 60-card Marnie representative, matchup tree, bundle, one-shot authorization, and label. Milestone packaging, quota waits, spacing waits, and uploads never freeze, stop, pause, select, or block Marnie training. The ordinary separately checksum-bound terminal completion submission remains required and is not replaced by a milestone snapshot. | Staged during the active Marnie H10 bootstrap without interrupting it. Install the idempotent managed watcher now, keep it fail-closed until the Marnie runtime registration and durable commits exist, and activate it with managed H10 RL. |
| 108 | 2026-08-02 | Marnie's active practice/public mix and formal premium holdout must use the newly frozen final-format H10 Alakazam refresh, checkpoint `sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92`, instead of the historical V5 Alakazam opponent. Preserve the historical V5 package and prior results as immutable evidence, but supersede it in Marnie's active opponent roster. Both training practice and the additive S+ holdout must resolve the same checksum-bound H10 package; research controls remain unchanged. | Activated at the durable iteration-1 commit boundary. The revision-109 runtime uses scoped frozen registry `sha256:c3125064a4bb9a806fd65237eceb0fb9bf0dbe8da561971697f296b8539a622c`; iteration 2 assigns the exact H10 Alakazam opponent 270 practice games with a 135/135 seat split. Historical V5 artifacts remain immutable and research controls are unchanged. Final activation receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-h10-alakazam-opponent-activation-r109.json`. |
| 109 | 2026-08-02 | Marnie must proceed from iteration 1 to iteration 2 regardless of whether iteration 1's candidate policy is promoted: a rejected or unsafe candidate falls back to the last safety-approved learner, but does not stop the loop. Raise Marnie's formal premium skill-weighted win-rate requirement from `0.50` to `0.80`; retain the independent `0.50` confidence-lower guard, individual-opponent floors, official-control guard, complete 4,250-game audit, iteration-5 terminal floor, and iteration-15 ceiling. The 80% terminal strength gate may not be misapplied to block the iteration-1→2 training transition. | Activated at `2026-08-03T00:27:37.044256+00:00` from immutable iteration-1 commit `sha256:b68e1a22d9e4343193ad54889dfe6a94ea7a2ec7dcb7e49895c88146e244f364`. The candidate was not promoted to champion, but passed the safety pipeline and became the continuous learner `sha256:6d310f727d0b76dea4e600e7c8fb5ca08f77c584cdf6fc458177ed34d90ae1c5`; iteration 2 is actively collecting from that learner. The terminal gate now has distinct identity `specialist-strong-public-roster-sw80-at-iter5-v1+frozen-specialists-r14-r109`, requires `0.80` skill-weighted win rate and independent `0.50` confidence lower bound, retains the 4,250-game audit and iteration 5/15 floor/ceiling, and does not block early continuation. Receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-h10-alakazam-opponent-activation-r109.json`. |
| 110 | 2026-08-03 | Self-play, public-mix, promotion, and formal-holdout collection may not retain exhausted simulator processes while completed results are serially compacted. The observed Marnie iteration-4 tail near `20 s/game` and roughly eight hours is rejected as a memory-pressure/ingest failure mode, not an acceptable simulator rate. Beginning with Marnie iteration 5, `POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN=1` is a launch requirement: after every local/remote producer has finished and every result/done record is durably queued, release the phase-owned local pool before consuming the remaining bounded RAM/disk result buffer. Apply the same producer lifecycle to scheduled and legacy additive dispatch and to self-play, public mix, promotion, research-control, and formal holdout waves. Preserve all exact-row, seat, checksum, and retry audits; never discard or recollect a durable buffered result merely to improve the display. | Activated at the exact immutable iteration-4→5 boundary. The recovered transaction preserved the already-running Latest20 trainer PID `280792`, advanced only the stale gate watcher to PID `311938`, restored all three persistent runtime drop-ins, and sealed activation receipt `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latest20-runtime-activation-r109.json` (`sha256:fc23eed0dabb0e42e80eadfad74f6ec6975dd68e4f3de63eda9f3accd6bcadf6`). Iteration-5 self-play emitted producer-complete telemetry releasing all 96 phase-owned local children before the compaction tail; host available memory rose from about 12.4 GiB to 37.2 GiB before the public-mix pool started. Public mix is active under the same implementation; later promotion, research-control, and formal-holdout waves inherit it and remain subject to their own live telemetry audit. |
| 111 | 2026-08-03 | Reweight the active Marnie public-mix and formal premium-holdout opponent contract by architecture and provenance. Every eligible non-active H10 specialist is tier `S` with canonical weight `2.0`; every other frozen specialist is tier `A` with weight `1.0`; every remaining public opponent is tier `A` with weight `1.0`. Preserve the exact 17-opponent checksum roster, 250 games per opponent, 4,250 formal games, 125/125 seats, research controls, and independent `0.80` skill-weighted / `0.50` confidence-lower terminal guards. The current iteration-5 wave and its receipts remain immutable under revision 109. | Activated for iteration 6 at `2026-08-03T18:24:23.998816+00:00` from exact iteration-5 commit `sha256:c22dab55963009126d6669832335d24e4abdeb54fbc738939a81923e684204e9`. Receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-opponent-tier-activation-r111.json`, `sha256:1d866f198c9e02f51dbd91f2cce6847843011a36457c9634f00a69405afed080`. |
| 112 | 2026-08-03 | Eliminate the scheduled fleet tail that preassigns the final public-mix/holdout games to remote-owned chunks after Blackwell finishes. For future scheduled additive waves, enter tail work-stealing with 20% of the wave unclaimed: return excess endpoint-owned reservations to the shared exact-job pool, retain at most one execution wave per remote endpoint, force one-game remote claims, and keep the 96-worker Blackwell pool eligible until global completion. Results continue to stream game-by-game; network/RPC batching is not introduced. Preserve every job, seed, seat, replay row, checksum, retry, and producer-complete drain audit exactly once. | Activated for iteration 6 at the exact revision-111 boundary with implementation `sha256:9c231047f5edc04a9421993edbca7e70ebad418fd67841707b88400d2cc94053`. Receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-tail-work-steal-activation-r112.json`, `sha256:2226ecbda30d652a6d30f5b93c756732d8819a46331729073ca94575e5bb548e`. |
| 113 | 2026-08-03 | Extend the active final-format Marnie's Grimmsnarl ex refresh through durable zero-indexed `iter_00020`, never collect `iter_00021`, then freeze/register that exact checkpoint under the existing measured-pass-or-explicit-ceiling-authority rule. After Marnie completes, prepare and train a new separately versioned H10 Crustle specialist before population training. This supersedes revisions 61 and 63 only where they excluded Crustle from specialist training: preserve Crustle's fixed Router Format 6 slot and reuse `pilkwang-meta-20260708` only as an immutable inference baseline/practice opponent. The new Crustle run must resolve the then-current H10 architecture, all-head bounded option-conditioned fusion, matchup adapters, current-deck guide, exact representative/corpus, terminal preflight, freeze, registration, and submission contracts. | Marnie's iteration-20 ceiling activated for iteration 6 at the exact revision-111 boundary; receipt `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-iteration20-activation-r113.json`, `sha256:9e45f35117c8c0edd32ba40a7f67e2bc9fecde1ab2ea1085f047881b032d4199`. The stale live completion override that bypassed Crustle was repaired without restarting Marnie; all five Crustle units are loaded and dormant, Marnie completion now targets Crustle bootstrap, and population release requires Crustle completion. Handoff-stage receipt: `/home/inzi/poke-bot-agent/outputs/state/post-marnie-crustle-r113-handoff-stage-v1.json`, `sha256:e1ed637c2262e749fb7ca45867c6618ad172cd77e204580f0b7658ceca6f2ccc`. Crustle gradients remain forbidden until Marnie's immutable iteration-20 completion handoff and the corrected v2 corpus validation. |
| 114 | 2026-08-03 | For the active Marnie lineage, stage a neural-only action-conditioned latent-lookahead policy aid; prohibit MCTS, beam search, and competition-time simulator search. Use the best checksum-exact heldout parent and a zero-safe migration. Latent action authority remains disabled unless paired ladder-proxy, fixed-holdout, causality, step-zero parity, latency, and replay-isolation gates all pass. Further policy ideas and fresh Kaggle replay evidence are analysis-only until separately canonicalized. | Zero-safe shadow staged at the exact iteration-5→6 boundary from protected parent `sha256:563277f8af8a581dd6277a5704d51057ba45172f58f7eed956d6e13271c43e26`, with inherited tensors bit-identical, zero initial policy drift/KL, and authority off. Receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latent-lookahead-r114-stage.json`, `sha256:3b31141e66353afcc24bd2ec638a4343514915816b18d177c76d9e2d003116d2`. Revision-116 training is nonblocking and all authority gates remain pending. |
| 115 | 2026-08-03 | Create exactly one no-training Kaggle experiment by copying the immutable final-format H10 Alakazam completion bundle (`sha256:e596630536d5052ae172ba2a42d72023709eba8b98c17a47243d1275b33a5b75`, model `sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92`) and replacing only `deck.csv` with the owner's exact corrected 60-card list, including 3 Dudunsparce and 3 Alakazam. Submit one first-preferring copy labeled exactly `new list experiment, h10 alakazam, no train`. This experiment does not train, freeze, register, promote, change the canonical Alakazam representative, consume a normal specialist copy, alter the active selector, or affect Marnie. | Activated as Kaggle submission `55217604` at `2026-08-03T17:08:25.873000+00:00`; evaluation was pending at activation. The derivative bundle is `sha256:5696a66f47f1f9fd7999cf70b5fc93908e97537a76678a728a228339f526a5b0`; all 153 archive files match the immutable source except `deck.csv`, the isolated 60-card neural-engine smoke passed 80 steps, and the digest-and-label-bound one-shot guard returned 0 and consumed its authorization. Receipt: `state/alakazam_h10_no_train_list_experiment_r115.json`. |
| 116 | 2026-08-03 | Marnie iteration 6 must begin the implementation phase of the selected revision-114 policy improvement, not merely materialize dormant tensors. After the exact iteration-5 commit, launch a separately versioned shadow update for the neural action-conditioned latent-lookahead plus bounded policy-aid challenger from the checksum-exact heldout parent. Train only from training-eligible Marnie replay; Kaggle and every formal evaluation remain evaluation-only. Keep latent action authority and serving eligibility disabled, preserve the existing iteration-6 lineage as nonblocking fallback, and do not let challenger failure or resource unavailability delay revisions 111–113 or ordinary Marnie training. The other report proposals remain analysis-only until separately canonicalized. | Shadow update completed at `2026-08-03T18:27:05.738886+00:00` while ordinary iteration 6 remained live. It used 256 uniformly strided games from the exact 8,192-game training-eligible iteration-5 shard, produced 128 updates with finite nonzero gradients across all 12 latent tensors, kept protected parent/base tensors bit-identical, used isolated optimizer state, and retained action/serving authority off. Receipt: `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latent-lookahead-r116-shadow-train.json`, `sha256:6bbbae1e93cc007f5558780c819a3524144b9906f0f08eadbd83a29a477bd1aa`. Static validation then proved exact authority-off policy parity, bit-identical protected tensors, training-replay isolation, finite nonzero bounded aid (`|aid|max=0.020941 < 0.25`), and distinct option/state conditioning; receipt `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latent-lookahead-r116-static-validation.json`, `sha256:4bca1e99166468757e2d5191f8a495b882c04988d00fa2f619eba4dfe7153f88`. Dynamic causal replay, paired ladder-proxy, fixed-holdout, full-policy latency, protected-parent drift/KL with authority, and exact-gate nonregression remain mandatory before any action authority. |
| 117 | 2026-08-03 | Keep Marnie's intentionally difficult `0.80` terminal specialist-strength gate separate from revision-114/116 latent-head action authority. The terminal gate governs completion, freezing, and the move to Crustle; it must not gate whether the new action-conditioned latent-lookahead/policy-aid head can help Marnie during iterations 7–20. Latent authority instead requires its own checksum-bound, parent-relative safety suite: causal replay, step-zero parity, replay isolation, bounded full-policy latency, bounded authority-on KL/drift, paired ladder-proxy nonregression, fixed-holdout nonregression, and exact-gate nonregression. On a complete pass, merge only the validated latent tensors into the then-current shape-compatible Marnie learner and enable their bounded action route at the next clean commit boundary, earliest iteration 6→7. Passing this suite neither completes Marnie nor advances decks; failure or lateness keeps the head shadow-only and ordinary Marnie training continues. | Staged while iteration 6 remains healthy. The gate/evaluation and checksum-bound merge/activation chain must be implemented and validated without interrupting iteration 6. |
| 118 | 2026-08-03 | Stop the uncommitted Marnie iteration-6 attempt and apply the policy/head changes previously staged for iteration 7 to the restarted iteration 6 now. Activate the trained revision-116 action-conditioned latent-lookahead policy aid with its bounded `0.25` authority route, preserve the exact iteration-5 learner as rollback parent, retain the unchanged `0.80` terminal deck gate, and treat the revision-117 parent-relative suite as a post-activation monitor with fail-closed rollback rather than a prerequisite. Carry every existing H10/Fusion-v3 head, learned route reliability, guide weight `0.05`, Router Format 6 bank, and matchup-fit workflow into the restarted run. Publish this as Accepted Policy Generation 15 because generations/attempts 10–14 already exist and may not be overwritten. | Activated at the restarted iteration-6 boundary at `2026-08-03T19:17:58.585132+00:00`. The authority-on checkpoint is `sha256:9ff2a8bcaf9ee51db1f6bb7dd86fb5d480a1737d80859140323f9b6fcab36cc5`; activation receipt `/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-latent-lookahead-r118-activation.json` is `sha256:9243ccd6632785e6c01bd7a3380b8c57eda0c9b5d0c4db0abddc2b87deaba22d`; clean-boundary migration `migration_0008.json` is `sha256:934fd50cdd258cb92c932002835a73d87e8c3a3b036bd349d1267baee3097fc2`. The managed run is live with Generation 15 as both champion and learner; the exact protected generation-9/iteration-5 parent remains rollback and heldout comparator. |
| 119 | 2026-08-03 | For self-play only, revise the 20% scheduled tail so Bert receives no new claims, Elmo retains at most one executing plus one queued GPU wave with first remote claim, and Blackwell's 96 local workers remain eligible. Restore the already-frozen Dragapult/Dusknoir and Hammer-Pult executables to Marnie's public practice without restoring deleted Dragapult specialist targets. | Activated for the restarted iteration 6 through the managed runtime environment. Live telemetry remains subject to review; any later tuning must preserve exact jobs, seats, seeds, and non-interruption of a started collection. |
| 120 | 2026-08-03 | Build observed-list Marnie archetype-family generalization and a separately typed existing-head loss contract. Keep the current exact 60-card list as the singular package/measurement deck; use no synthetic variants, no new policy heads/routes, and no Kaggle-derived training or tuning. Require at least 12 non-package swap-distance clusters, cluster-stable train/dev/locked splits, exact provenance, family-macro replay weighting, capability-masked losses, isolated same-parent antithetic SPSA, and the complete locked statistical/safety gate. Activate the selected family sampler and loss vector atomically only after a checksum-exact successful iteration-9 Kaggle upload receipt and at the first later clean pre-collection boundary. | Staged after revisions 118–119. No iteration-9 successful-upload trigger currently exists; active iteration 6 and its exact-list recipe remain unchanged. Missing evidence or any failed/inconclusive gate fails closed. |
| 121 | 2026-08-03 | Fix self-play remote-only tails before the revision-120 family activation work can affect production. Reject pre-reserving hundreds of a 1,024-game wave to remotes. Cap each endpoint's controller-owned steady-state queue at one execution wave, enter the self-play drain at 35% unclaimed, give Bert no new tail claims, and retain only one executing Elmo wave with no queued tail wave. Preserve Blackwell at 96 workers and preserve all exact jobs, seeds, seats, and replay rows. | The user explicitly authorized stopping the uncommitted iteration-6 attempt. The managed Marnie service was stopped through systemd; implementation and focused tests were staged before managed restart. Revision 120 remains later and runtime-inert. |
| 122 | 2026-08-03 | After five repeated final-return stalls, keep Elmo and Bert in self-play but restrict them to an early 16+4 execution wave: enter the no-new-remote-claim tail while 75% of jobs remain, cap the maximum remote-owned tail at 20, reserve control-plane connection headroom, and compress large JSON trajectory frames for the 100-Mb return path. Blackwell's exact 96-worker pool owns the long drain; public mix and evaluation retain their full remote caps. | Authorized for immediate recovery of the stopped, uncommitted iteration 6. Clear the abandoned remote generation, deploy the frame codec to both managed remote workers before the controller, activate with `POKEBOT_SELF_PLAY_LOCAL_ONLY=0`, and verify no more than 20 remote claims plus successful compressed returns. |
| 123 | 2026-08-03 | Supersede revision 122's inferred 16+4 self-play cap. Keep the full 36-worker Elmo and 16-worker Bert execution targets for self-play, public mix, and evaluation. Fix the return path with compressed exact trajectory frames and truthful telemetry that separately reports live sockets and outstanding remote-owned games; do not hide a return/ownership defect by lowering usable fleet capacity. | Staged without restarting the active recovered iteration-6 attempt. The currently executing attempt keeps its already-loaded process environment; the no-cap runtime and corrected telemetry activate at the next safe managed process boundary. Public mix remains uncapped throughout because the discarded cap was self-play-scoped. |
| 124 | 2026-08-03 | At the owner's immediate stop-and-full-resync boundary, correct the two verified self-play fleet defects without lowering capacity: negotiate compressed frames before either peer emits them, and force exactly one privately claimed self-play game per request socket so all 36 Elmo plus 16 Bert execution workers can work concurrently. Keep both remotes eligible until exactly 20 shared jobs remain; only then stop new Bert claims while all already-owned remote games finish and return. Dashboard telemetry must distinguish live request sockets from exact outstanding remote-owned games and split the latter by Elmo/Bert. | Trainer and its relaunching gate handler are runtime-masked during repair. Codec negotiation and complete-trajectory return canaries passed on both remotes; focused scheduler/wire tests passed. Activate only after the checksum-identical three-host resync, managed-unit preflight, and first live dispatch prove 52 sockets, 52 initial outstanding games, nonzero work on both remotes, and successful decrements on returned trajectories. |
| 125 | 2026-08-03 | Do not repeat a multi-minute remote checkpoint load or full SMB payload checksum scan when the exact digest is already resident and a fresh strict health proof shows the controller and every leaf healthy, identity-exact, and pinned. Seed the process-local digest-addressed staging map from that proof. Production performs a verified stage/reload for incomplete or mismatched health and every genuinely new digest. A diagnostic canary must refuse implicit reload and require explicit `--force-reload`, so merely checking health can never cause another 141-second operation. | Authorized for immediate activation while the trainer remains stopped at the revision-124 repair boundary. Production hard-gate reuse is implemented; canary reuse and regression coverage must pass before managed restart. |
| 126 | 2026-08-03 | Complete revision 120's local, runtime-inert family-generalization implementation and its fail-closed trainer boundary hook. Preserve the singular package deck, Router Format 6 identity, all serving/gate settings, and the exact-list production recipe until every upload, manifest, shadow-study, and clean-boundary receipt exists. | Implemented and locally validated; not activated. No real iteration-9 upload trigger or activation-ready observed-family manifest is present in this repository. |
| 127 | 2026-08-03 | Eliminate false full-fleet reservations caused by serialized historical-checkpoint SMB verification and serial initial socket opening. Verify missing Elmo receipts beside TrueNAS storage with a root-bounded capability, reuse checkpoint identity by immutable digest, cache local hashes only across unchanged path/size/mtime/inode identity, and open initial Elmo/Bert request sockets concurrently under bounded LAN deadlines. Preserve exact games, seeds, seats, results, the 96+36+16 hardware profile, and revision-124's 20-game tail rule. | Activated for the restarted iteration 6. Receipt `state/remote_scheduler_checkpoint_activation_r127.json` (`sha256:24baf059954708861a18c3598f6b9acd9d3c2ee4feeef50a5fa99a047f512b62`) binds the storage-local 0.075-second iter-3 digest proof, one successful real Elmo self-play canary, durable H10 worker image `sha256:0bcf2305438f8feecd9420cc37af8da4e3a2d81986e112597ad38fbe1e3f1aa3`, 0.199-second 50/50 clone fan-out, live 36+16 admissions, managed PID 1267326, migration 0018, and current dashboard integrity. |
| 128 | 2026-08-03 | Reconcile the goal gateway after the managed iteration-6 resume. Correct the compatibility projection to the checksum-exact Generation-15 runtime registry, record the completed self-play to public-mix transition, and preserve the current scheduler and 20-game self-play tail behavior exactly. | Activated as metadata-only state reconciliation. Receipt `state/marnie_iteration6_resume_snapshot_r128.json` (`sha256:68cfe56e687e4a322cac69c609038eb915f94d160a094811c44f0472ad3748de`) binds PID 1267326, checkpoint `9ff2a8bc…`, registry `72b61af5…`, the live public-mix phase, full 96+36+16 fleet, and current dashboard integrity. No managed service was restarted. |
| 129 | 2026-08-03 | Reconcile the post-Marnie Crustle pre-stage after the v2 guide corpus completed and imported. Validate every imported corpus digest and keep the dormant managed chain authority-off until Marnie iteration 20; require a then-current H10 registry rebind at the actual handoff. | Corpus blocker resolved by `/home/inzi/poke-bot-agent/outputs/state/post-marnie-crustle-r113-handoff-ready-v2.json` (`sha256:ccd18d1b33feb2bc05ce5491dee51f95ee95db301da8433c43e96d6b123f978b`). It binds 26,932 games, 1,428,142 decisions, 33,620 guide rows, all five loaded/inactive Crustle services, and active unchanged Marnie PID 1267326. Marnie iteration-20 completion and the then-current H10 registry rebind remain hard blockers; no service was restarted. |
| 130 | 2026-08-03 | Make the successful checksum-exact Marnie iteration-9 Kaggle submission the immutable last old-system boundary. The next Marnie training collection must use the new system; this boundary is not mutable. | Staged while iteration 6 remains healthy. Stage receipt `state/marnie_new_system_boundary_stage_r130.json` (`sha256:98e76bcde0277e36f2790ce1933ff38f3e57e16948e4c7bd1999ab714d1fe709`) binds the contract and unchanged PID 1267326. Prepare and validate every new-system activation artifact before the trigger. After the successful iteration-9 upload, missing or inconclusive activation evidence pauses before collection instead of continuing the old system. No current iteration, scheduler, selector, or managed service is changed by recording this boundary. |
| 131 | 2026-08-03 | Reconcile the resumed goal and arm the revision-130 hook at the next clean commit, without changing the immutable post-iteration-9 transition or any scheduler behavior. | Armed under `pokebot-marnie-family-hook-boundary-r130.service`, waiting for the exact iteration-6 commit before a managed restart ahead of iteration-7 collection. Receipt `state/marnie_new_system_hook_monitor_armed_r131.json` (`sha256:193ec20f2e720ff7e989734abf940964227e27467cc4103a2996e6d2d1ab3152`) binds active PID 1267326, zero restarts, pending monitor PID 1374045, and unchanged 96+36+16 fleet/tail behavior. The successful iteration-9 upload remains the final old-system boundary; new-system training does not begin early. |
| 132 | 2026-08-03 | Reaffirm as immutable that Marnie remains on the current system through iteration 9 and its successful checksum-exact Kaggle submission; only the first later collection may begin new-system training. Prepared or passing artifacts cannot activate it early. | Canonical immediately as an owner boundary clarification. No current training, scheduler, fleet, hook, or self-play behavior changes. Post-upload evidence still fails closed to a pause if activation is not ready. |
| 133 | 2026-08-03 | Record completion of the already-authorized revision-131 boundary hook after the exact iteration-6 commit. This is operational reconciliation only and does not move the immutable post-iteration-9 activation boundary. | Activated at `2026-08-03T23:57:16.456805+00:00`. Managed PID `1267326` was replaced by PID `1471079` between iteration 6 and iteration 7; receipt `state/marnie_new_system_hook_activation_r130.json` is `sha256:f26796f6776c48cdac4e63e6e5ddd7045bc747c59a3dec1fbf8acf54b455e9c1`. Iteration 7 started with only the dormant trigger/request paths loaded; family sampler/loss authority remains off. Scheduler, 96+36+16 fleet, and exact 20-game tail rule are unchanged. |
| 134 | 2026-08-04 | After the successful checksum-exact Marnie iteration-9 upload activates the new system, run exactly 25 expert bootstrap epochs over the checksum-pinned 73,082-record/6,828,373-decision Marnie's Grimmsnarl ex corpus, submit that exact bootstrap checkpoint, and begin new-system self-play only after its successful checksum-exact upload receipt. | Staged and runtime-inert while iteration 7 remains healthy on the current system. A missing, failed, or pending bootstrap/submission receipt pauses before self-play and never authorizes another old-system collection. No live scheduler, fleet, or tail behavior changes. |
| 135 | 2026-08-04 | Give public expert replay acting seats from checksum-pinned PTCGReplay top-100 pilots higher post-upload-bootstrap importance, increasing only with adequate same-corpus pilot support: `1.5x` at 1--31 games, `2.0x` at 32--127, `3.0x` at 128--511, and `4.0x` at 512+; unmatched seats remain `1.0x`. Require an exact `episode_id + seat + TeamNames[seat]` join and immutable snapshot/index receipts. | Ready and runtime-inert for the revision-134 post-iteration-9 bootstrap only. All 73,082 pilot rows resolved with zero unverifiable records; 33,156/65,774 training rows match the pinned top 100. The immutable weight index is `sha256:d0c978ba12c0e758747a5d1ea185f668b579a0905ef9bed2609d998febe2ec05`, with tier counts `1.0x=32,618`, `1.5x=206`, `2.0x=1,648`, `3.0x=4,120`, `4.0x=27,182`. It does not alter current iteration 8, public collection, gates, or Kaggle evaluation replay isolation. |
| 136 | 2026-08-04 | Supersede revision 135's upper support bands after verifying substantial same-training-split evidence: retain `1.5x` at 1--31, `2.0x` at 32--127, and `3.0x` at 128--511; use `4.0x` at 512--1,023, `5.0x` at 1,024--2,047, and a bounded `6.0x` at 2,048+. Unmatched seats remain `1.0x` and validation remains unweighted. | Ready and runtime-inert for the revision-134 post-iteration-9 bootstrap only. The pinned split has 22 top-100 pilots at 512+, nine at 1,024+, and four at 2,048+, supporting the higher ceiling without amplifying sparse pilots. Immutable r136 index `sha256:d7a0e6d60ec02edf2fa831eea7231bf663bf94b98105f28f31046566f8df26da` has effective mass `183,984` and tier counts `1.0x=32,618`, `1.5x=206`, `2.0x=1,648`, `3.0x=4,120`, `4.0x=9,697`, `5.0x=8,297`, `6.0x=9,188`. No active iteration, scheduler, fleet, tail, or evaluation behavior changes. |
| 137 | 2026-08-04 | Raise the well-supported top-100 pilot bands again: preserve all tiers through 512--1,023, increase 1,024--2,047 from `5.0x` to `6.0x`, and increase 2,048+ from `6.0x` to a bounded `8.0x`. Validation and unmatched seats remain `1.0x`. | Ready and runtime-inert for the exact 25-epoch post-iteration-9 bootstrap. Nine pinned top-100 pilots have at least 1,024 training games and four have at least 2,048, so only large-sample pilots receive the higher bounds. Immutable r137 index `sha256:50928fe4ab7f467ebf03a7c555073a48df5bf62176cf41cf825a9601933b86a2` has effective mass `210,657` and tier counts `1.0x=32,618`, `1.5x=206`, `2.0x=1,648`, `3.0x=4,120`, `4.0x=9,697`, `6.0x=8,297`, `8.0x=9,188`. No active iteration, scheduler, fleet, tail, or evaluation behavior changes. |
| 138 | 2026-08-04 | Raise only the large-sample top-100 pilot bands again: `7.0x` at 1,024--2,047 exact training games and a bounded `10.0x` at 2,048+, preserving all lower tiers, unweighted validation, and `1.0x` unmatched rows. | Ready and runtime-inert for the exact 25-epoch post-iteration-9 bootstrap. The highest four pilots each have 2,094--2,539 exact training games. Immutable r138 index `sha256:1d9ee77af9b9d5f916037b69475471c0a20602c6ea4ba186b59deb8fc07371dc` has effective mass `237,330`, an analytical effective sample size of `36,169.67` (`54.99%` of 65,774 training games), and tier counts `1.0x=32,618`, `1.5x=206`, `2.0x=1,648`, `3.0x=4,120`, `4.0x=9,697`, `7.0x=8,297`, `10.0x=9,188`. No scheduler, fleet, tail, label, action, or validation behavior changes. |
| 139 | 2026-08-04 | Activate Marnie's archetype-family sampler and typed family-loss vector under explicit owner ceiling authority after the valid two-round study remained inconclusive. Preserve the failed measured study and select exact round-1 `plus`; never label it a measured pass. | Authorized for immediate checksum-bound managed activation at the existing status-75 post-upload boundary. Atomic migration, exact-parent/bootstrap continuity, epoch-25 submission, upload-before-iteration-10, and the post-activation status-78 monitor remain mandatory. |
| 140 | 2026-08-04 | Permanently retire Marnie's guide for this lineage so learned policy and fused heads can improve beyond the hand-authored scaffold. Set its weight to exactly `0.0` for bootstrap and every later Marnie update; keep historical guide artifacts audit-only. | Activated at the existing post-iteration-9, pre-bootstrap receipt-backed boundary. Resume the exact weighted 25-epoch bootstrap from the uploaded parent with all non-guide objectives unchanged. Submission/activation repair receipt: `state/marnie_guide_retirement_submission_chain_r140.json` (`sha256:47758999c5290f806defa0401c1b6e3431e112032d20c7a165e4c75e6211ac24`). |
| 141 | 2026-08-04 | Clarify that Marnie's retired guide may remain as an optional offline shadow diagnostic, while retaining exactly zero training and runtime authority. It cannot influence loss, gradients, fusion, actions, serving, checkpoint selection, gates, submission, self-play authorization, or blocking behavior. | Active immediately as a non-authoritative contract clarification. Missing, invalid, or failed shadow evidence is nonblocking. Shadow receipt: `state/marnie_guide_shadow_non_authority_r141.json` (`sha256:fe0d43ce38eab29c6ee49a5b6047b8b4059fb50dca10b3e832612f100379222b`). The epoch-1 family-residual validator repair resumes the immutable guide-off checkpoint without retraining under `state/marnie_postupload_epoch1_recovery_r141.json` (`sha256:c08284d1d53f39bd804f588c172a768d99b23888ee9e1a61653ed428197f4fe7`), and its end-to-end submission/activation binding is `state/marnie_epoch_recovery_submission_chain_r141.json` (`sha256:401344d07388234722b1dd3fdbcc6805af2aa499b3c80280ea09ef6cdc590f11`). Dashboard deployment: `state/marnie_guide_shadow_dashboard_r141.json` (`sha256:5d14e5ef522a39684c1873aa81ae7572f1dcd9a29a3ae7fc20d4df11885816b7`). |
| 142 | 2026-08-04 | Repair the dormant iteration-10 registry precedence defect: merge the active family registry with Marnie's zero-authority shadow-guide contract, preserve every non-guide field, and ensure both post-monitor continuation and family rollback remain guide-off. | Activated as a next-start-only managed overlay without interrupting the active bootstrap. Receipt `state/marnie_family_guide_shadow_runtime_r142.json` (`sha256:1ea409b8315b8d5fe1c3b0ad2d2e5245f4c7e3ceb2907f3e59a4fd7f343fc88a`) binds family registry `sha256:22f53258…`, retired parent `sha256:6960318b…`, merged registry `sha256:88b10545…`, and the final drop-in. End-to-end activation/monitor chain: `state/marnie_family_guide_shadow_chain_r142.json` (`sha256:5fa2a96582d454a2cdf80e667bcfd6657f121af76e0577759dbd67bcff8da320`). Dashboard receipt: `state/marnie_family_guide_shadow_dashboard_r142.json` (`sha256:59bc2843c5017c096cfd4a3a1e16e7c4406157d9306eea1dedd8dc47b02e5faa`). |
| 143 | 2026-08-04 | Repair the dormant Marnie→Crustle boundary so completion executes the actual transaction and Crustle binds the exact handoff-time H10 registry instead of the stale pre-Generation-15 r113 registry. Strip Marnie-only family and guide metadata before creating the Crustle runtime. | Activated for the dormant boundary without interrupting the active bootstrap. The effective completion service has one real completion `ExecStart`, the current managed registry is checksum-bound into the completion receipt, all five Crustle units remain inactive, and focused tests pass. Receipt: `state/post_marnie_crustle_runtime_rebind_r143.json` (`sha256:ddd25f2ad7c42182cec5e858ec8c48377b9ca1ff00fe9ebd0e0fc5ff652b0539`). |
| 144 | 2026-08-04 | Repair the dormant post-Crustle population materialization boundary. Preserve the immutable 14-row historical frozen-specialist registry; form the 15-member trainable population from those 14 identities plus the three checksum-bound refresh rows, where refreshed Alakazam and Marnie replace their current versions and newly completed H10 Crustle adds member 15. Bind Crustle's exact final-format bundle, expert manifest, matchup tree, and runtime registry into its refresh row. The public Crustle baseline is not a trainable member and is not selected history. | Activated for the dormant boundary without interrupting the active guide-off bootstrap. Population preparation fails closed unless the union is exactly 15, all three refresh artifacts are checksum-exact, and Crustle has no historical/public training entry. Focused regression suite: 16 passed. Receipt: `state/post_crustle_population_materialization_r144.json` (`sha256:fb11c4a211d5598a35d9ae68b9ad5caa59a59a8119827028b85bc5159a87a8c9`). |
| 145 | 2026-08-04 | Make Marnie's optional guide-shadow bypass finite as well as non-authoritative: inactive guide/setup placeholders must be exact zero-gradient tensors even when packed option padding contains negative-infinity logits. | Repaired in the canonical training tree and active deployment without interrupting iteration 10. The active process retains finite non-guide gradients and zero guide influence; the repaired implementation loads at the next clean managed interpreter start. Regression coverage requires finite expanded resident loss and guide loss exactly zero, and rehearsal now fails before backward on any non-finite total. Receipt: `state/marnie_guide_shadow_finite_bypass_r145.json` (`sha256:4ddc5f2e8a449a880e0c3e43a5e808f0ddac1dc8dff27be2c45def7c44373b15`). |
| 146 | 2026-08-04 | Keep Marnie's retired guide from blocking RL candidate persistence: legacy/guide-off mode must not request a strategic-curriculum checkpoint record. | Repaired with one shared rehearsal/RL guard, synced to the active deployment, and loaded by the managed status-1 restart. Iteration 10 reused its immutable collection and rehearsal receipts without recollection. Focused strategic tests and an end-to-end `rl_train_step` test pass. Receipt: `state/marnie_guide_off_rl_save_recovery_r146.json` (`sha256:aa032e7f2e8e512c2c712bf1d7d72c166899545fe6d54c3f472fe901e56a12dc`). |
| 147 | 2026-08-04 | Keep Marnie's family replay sampling checksum-stable during recovery-only source migrations by seeding from the sealed collection design fingerprint. | Activated in the managed iteration-10 recovery without recollection or rehearsal retraining. Exact source seats remain 4,096/4,096, the immutable replay projection is reused, seven focused tests pass, and training advanced into baseline preparation with the guide still shadow-only and nonblocking. Receipt: `state/marnie_family_replay_seed_recovery_r147.json` (`sha256:4379636ca9e012dab0fbaea0fb7d4f00b1f097a9228f60f5f3282bf327b09824`). |
| 148 | 2026-08-04 | Complete the mandatory first-new-system family monitor after immutable iteration 10 and preserve Marnie's retired guide as optional shadow-only evidence through either monitor outcome. | The fresh 4,284 locked-pair plus 1,020 package-pair audit required rollback: package delta lower bound was `-0.0825242791`, with zero invalid games and passing causality/latency checks. The resolver restored the exact iteration-7 heldout parent before iteration-11 collection. Monitor `sha256:84887e6b…`; resolution `sha256:7a0533a9…`. |
| 149 | 2026-08-04 | Repair the family-rollback resume registry without weakening the rollback gate: preserve the exact rollback migration reason, rebind only the operational runtime root to the already-deployed post-upload tree that implements it, and retain guide weight `0.0` with no guide authority. | Activated before iteration-11 collection. Effective registry `/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/runtime/specialist_runtime_registry_h10_r149_family_rollback_guide_shadow.json` is `sha256:e28dfbde…`; repair receipt `sha256:c6961239…`. The failed family sampler is off and all non-guide fields other than the source runtime root are preserved. |
| 150 | 2026-08-04 | Reconcile the rollback parent's heldout evidence with its checksum-exact iteration-7 checkpoint before resume; do not allow stale iteration-10 evidence to block or misrepresent the restored parent. | Activated before iteration-11 collection. The 4,250-game iteration-7 heldout evidence and checkpoint both bind `sha256:f20efb20…`; migration receipts 0027/0028 are immutable. Marnie resumed managed iteration 11 under PID `138326`, zero restarts, full `96+36+16` fleet, and guide-shadow weight `0.0`. Receipt: `state/marnie_iteration11_family_rollback_resume_r150.json` (`sha256:4509f091fc62753aa4d4177d7696bc3041a5f46568872e821aa93610627a2e99`). |
| 151 | 2026-08-05 | Reconcile the mutable specialist and dashboard compatibility projections to the already-running iteration-11 family-rollback runtime without changing production. | Activated metadata-only while iteration 11 continued from self-play into public mix. The managed PID, selector, checkpoint, scheduler, fleet, guide-shadow contract, and failed-family rollback authority were unchanged; only stale iteration-8/self-play projection fields were corrected. Receipt: `state/marnie_iteration11_public_mix_state_reconciliation_r151.json` (`sha256:fea9c0edd4ad7bf5316d0303f43d29cf963efaf7f5f13a054cb8d6135520657a`). |
| 152 | 2026-08-05 | Render the active post-fleet Marnie refresh as a separately versioned iteration-20 boundary followed specifically by the new H10 Crustle specialist; do not append historical-roster unfinished counts to a refresh action. | Activated display-only through the managed Bert dashboard. The API reports the exact iteration-20/no-iteration-21/Marnie-to-Crustle contract and 19/19 current sources. Marnie training remained on PID `138326` without restart or scheduler mutation. Receipt: `state/marnie_dashboard_postrefresh_next_action_r152.json` (`sha256:4bd2f4246cf06e76ab9fefb67244a34306dbb2eb6f8b20cdaafa27d37aee4af1`). |
| 153 | 2026-08-04 | Clear superseded iteration-7 heldout evidence when activating the exact post-upload epoch-25 checkpoint; require a fresh exact gate for the changed learner. | Activated before iteration-10 evaluation. No heldout evidence transferred between checkpoint digests and guide authority was unchanged. Managed receipt: `/home/inzi/poke-bot-agent/outputs/state/marnie-r153-postupload-heldout-evidence-repair.json` (`sha256:8af9603909fa79af9b0b672f51b1faabd80cb6ef91c4bc8f36f94eece0eef377`). Historical evidence remains immutable after the later family-monitor rollback restored iteration 7. |
| 154 | 2026-08-05 | Repair the dormant Marnie-to-Crustle completion preflight so its absolute canonical completion script can resolve the checksum-bound managed-runtime-registry helper from any configured working directory. | Activated without restarting Marnie. The exact effective `ExecStartPre` now emits `FINAL_FORMAT_MARNIE_COMPLETION_OK` and resolves revision-149 registry `sha256:e28dfbde…`; 12 focused tests pass and all five Crustle units remain dormant. Receipt: `state/post_marnie_crustle_completion_import_repair_r154.json` (`sha256:93f470f0bfcd496ca53c072151f406a689b06b08eea83bab88b3deae49c42aac`). |
| 155 | 2026-08-05 | Preserve Marnie's retired guide as an explicit third dashboard state throughout live RL: `shadow_only_non_authoritative`, weight `0.0`, and no gradient, fusion, action, serving, gate, or blocking authority. Do not collapse that durable state into the generic `absent` fallback when the completed post-upload bootstrap service is no longer current. | Activated display-only without restarting or changing Marnie. The selector-owned snapshot and canonical copy are checksum-identical, focused regressions pass, the live API renders the explicit shadow state, and all 18 required dashboard sources remain current while iteration 11 trains. Receipt: `state/marnie_dashboard_guide_shadow_projection_r155.json` (`sha256:5227718eeddeed918b69dff95b64abf3bd497c31863d5c90cdda978c43de73fb`). |
| 156 | 2026-08-05 | Keep a promoted Marnie candidate's between-iteration checkpoint publication visible as active work. Treat the authoritative `BETWEEN_ITER_HARD_GATE begin` marker as `heldout:checkpoint_staging`, and keep that known bounded phase current for at most the existing 20-minute stall window while the managed service remains active; do not leave the completed adapter bar displayed as degraded during a required new-digest load. | Activated display-only without restarting or changing Marnie. The canonical and selector-owned dashboard snapshots are checksum-identical at `sha256:417383cf…`; 13 focused Marnie/dashboard regressions pass. Iteration 11 then completed the exact local+two-remote hard gate for promoted candidate `sha256:b1567e5c…` and entered the 4,250-game formal holdout under unchanged PID `138326`, zero restarts, with all 19 required dashboard sources current. Receipt: `state/marnie_dashboard_checkpoint_staging_projection_r156.json` (`sha256:8eb0715b0a9e21a65c007043540b27d44973927d0ba3b3a21bb10f0a533a0440`). |
| 157 | 2026-08-05 | Preserve every authoritative live `measure:*` phase against post-train checkpoint-staging inference. A current research-control or other measurement counter must remain the dashboard phase until runtime advances it; an older hard-gate marker cannot replace it. | Activated display-only without restarting or changing Marnie. Canonical and selector-owned dashboard snapshots are checksum-identical at `sha256:9598ca84…`; 13 focused regressions pass. Iteration 11 completed and committed checkpoint `sha256:b1567e5c…` after 4,250 formal games and 1,000 audited research-control games, then iteration 12 began under unchanged PID `138326`, zero restarts. The live dashboard reports iteration 12 `collect:public_mix` with 19/19 required sources current. Receipt: `state/marnie_dashboard_measurement_phase_projection_r157.json` (`sha256:5a134791cbf89288bde0c70ab3f74fa63fe35d7321711a2bb0b1dc3d8cfdd22c`). |
| 158 | 2026-08-05 | Preserve exact 8,192-game retention while preventing a single unusable public-mix record from forcing full recollection after only four replacement seeds. Extend only public-mix targeted replacement recovery to 32 deterministic disjoint lanes and report the exact missing schedule cells on final failure. | Staged source-only under running Marnie PID `1295559`; no restart or live-interpreter mutation. The first failed attempt remains quarantined at exact `1024/1024 + 7167/7168`. Active-deployment source `sha256:a2e03ec8…` matches canonical source and 96 focused recovery/scheduling tests pass. Activation is next managed interpreter start, followed by live exact-retention receipt validation. Receipt: `state/marnie_public_mix_exact_retention_recovery_r158.json` (`sha256:360a077e85bfa9f82cfa6f04ea1b9340c9ef88670867ac4999a99f4a948f445b`). |
| 159 | 2026-08-05 | Keep the active final-format Marnie handoff card on the exact iteration-20/no-iteration-21 → new H10 Crustle contract even when the generic historical handoff source still names Archaludon→Slowking. Do not append historical unfinished-roster wording to the refresh. | Activated display-only without restarting Marnie or the dashboard. Canonical and selector-owned snapshots match `sha256:209cc43e…`; 238 dashboard regressions pass; the live API reports Crustle, terminal iteration 20, forbidden iteration 21, and 19/19 current sources. Receipt: `state/marnie_dashboard_handoff_fallback_r159.json` (`sha256:81e2a62e7db522e0d423ca55a1f5172f628be9690da1d3bebb3310cc6d876d28`). |
| 160 | 2026-08-05 | For the dormant post-Marnie H10 Crustle bootstrap, use Crustle's guide only for epochs 1–10, then run an exact 25-epoch guide-zero expert refresh as epochs 11–35. Select the final bootstrap checkpoint only from epochs 11–35. Checksum-bind the newly completed Marnie/Grimmsnarl iteration-20 checkpoint into Crustle's expert/practice contract without relabelling its actions as Crustle expert actions. | Staged while Marnie iteration 19 remains active. This does not restart or mutate Marnie. Crustle bootstrap/register/launch must fail closed until the 10+25 schedule, guide-zero second phase, selection window, and completion-bound Grimmsnarl identity are implemented and validated at the Marnie iteration-20 handoff. |
| 161 | 2026-08-05 | Weight the exact Crustle expert corpus during all 35 bootstrap epochs, including both the 10 guide-active epochs and the 25 guide-free refresh epochs, using the checksum-pinned bounded top-100 pilot-importance policy already validated for Marnie. Keep validation unweighted and preserve every replay action and causal label. | Staged with revision 160 at the Marnie iteration-20 handoff. Bootstrap fails closed unless the Crustle-specific importance index matches the protected corpus and deterministic split, contains supported top-100 matches, and is recorded in every epoch, frozen provenance, and ready receipt. Active Marnie is unchanged. |
| 162 | 2026-08-05 | Supersede the Crustle guide-off refresh: keep Crustle's current-deck guide active for all 35 bootstrap epochs at its canonical held weight, with the weighted expert corpus applied throughout. Do not run a guide-free phase or turn the guide into direct action imitation; it remains a bounded strategic training influence while the learned heads and RL policy remain authoritative. Select the final checkpoint from the full 1--35 window and record the all-guide schedule in every epoch and completion receipt. | Activated before Crustle bootstrap training. The Crustle bootstrap launcher and registration validator now require `poke_bot.crustle_guide_all_epochs/v1`, guide-active epochs `[1,35]`, zero guide-free epochs, and revision 162. Existing Marnie completion and guide-retirement state are unchanged. |
| 163 | 2026-08-05 | Select the exact Marnie's Grimmsnarl ex iteration-9 milestone checkpoint `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381` (milestone copy 1/1, first-if-allowed) as the canonical Marnie freeze for Crustle training. Use it training-only; do not upload or submit it to Kaggle. Preserve the historical iteration-20 completion/freeze and all prior receipts byte-for-byte. | Activated at the Crustle pre-bootstrap boundary through `state/marnie-canonical-training-freeze-r163.json`; the Crustle practice/holdout anchor and staged opponent bind this checkpoint and bundle while retaining the immutable Marnie completion registry as the source runtime lineage. |
| 164 | 2026-08-05 | Keep Crustle's checksum-bound strategic current-deck guide active after its 35-epoch bootstrap as well as during it. The registered Crustle RL lineage keeps the bounded `0.05` strategic-directional guide loss and its learned-head curriculum; the guide remains training-only with zero serving, direct policy-imitation, runtime-logit, action, and gate authority. | Staged for Crustle's checksum-backed bootstrap-to-RL registration boundary. The active bootstrap is not restarted or modified. The registration fails closed unless it binds the same guide contract, directional strategic curriculum, and 19-head Fusion-v3 route contract. |
| 165 | 2026-08-05 | Make the Crustle guide exception persistent: do not enqueue generic guide-on/guide-off reviews, ramps, or decay for Crustle, and never reduce its registered strategic-directional guide weight from `0.05` automatically. Any future Crustle guide change requires a new explicit owner decision and a checksum-bound boundary receipt. | Staged for the same bootstrap-to-RL registration boundary. This corrects the generic prospective guide-policy inheritance without altering the active bootstrap or granting guide runtime/action authority. |
| 166 | 2026-08-05 | For Slowking research, learn the archetype from every confirmed public Slowking acting seat rather than requiring one exact 60-card fingerprint. All 768 currently recovered games across `vibechu` and `ShumpeiNomura` and all three observed list versions are policy- and strategic-learning eligible. Supply deck contents and legal actions to the learner; use exact-list identity only for optional conditioning, card-capability masking, drift analysis, and day/list-held-out evaluation. Reuse the existing expanded strategic heads, `setup_board_outcome`, `combo_state`, bounded option-conditioned decision fusion, expert feature/rehearsal path, and causal training-only Slowking guide. | Materialized as research-only contract `config/slowking_archetype_learning.v1.json`, validator/index code, and tests. This does not reopen the terminal failed experiment or grant training, checkpoint, selector, serving, registration, or submission authority. |
| 167 | 2026-08-05 | Crustle's strong-public / frozen opponent roster must retain two distinct Marnie specialists: (1) historical old-format Marnie `specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98` with package digest `sha256:ae9f3c31…` as tier A/1.0, and (2) H10-format Marnie `specialist-marnie-final-format-h10-f20efb20f5c3` bound to checkpoint `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381` with package digest `sha256:f7c25cfd…` as tier S/2.0. Do not collapse them into one row or reuse the historical digest on the H10 identity. | Activated at the Crustle RL preflight boundary through checksum-bound gate/frozen-registry rebind; receipt `/home/inzi/poke-bot-agent/outputs/state/crustle-dual-marnie-roster-repair-r167.json` (`sha256:86e4e6b6…`) with gate `sha256:271c7c56…` (18 opponents / 4500 heldout). |
| 168 | 2026-08-06 | Add the historical public Crustle-busting Lucario package `yaroslav-lucario-v2-crustle` (content digest `sha256:2738a2e4394155b0122eeaa68cec9bbe0cc7dbb4b79f5d055827778444b68bb3`) to Crustle's strong-public practice roster and formal premium holdout as tier A/weight 1.0 alongside the dual-Marnie A+S rows. This is an explicit owner retention of that public package despite frozen Lucario specialist supersession of other lucario externals. Expand the checksum roster from 18→19 opponents and formal games from 4,500→4,750 with exact 250/125/125 seats. Do not recollect iteration 5 or delete quarantine. | Staged for the next clean pre-collection / remaining-holdout boundary after the in-flight iteration-5 corpus restore. Live r163 gate/frozen registry digests remain bound to the restore fingerprint; staged artifacts are `runtime/final_format_crustle_gate_r168_lucario_a.json` (`sha256:611c67e6d4db1ae4995c307ec65180f49b4b2a9299c57c9eedcd66c7e8f1580a`) and `ops/frozen_specialist_registry_crustle_r168_lucario_a.json` (`sha256:7dc9768f8ec9b6a624519128d5fa202d51ad083f43d4d41023b4d7b10e803e32`). Receipt: `outputs/state/crustle-lucario-a-tier-roster-r168.json` (`sha256:325f55fde68272eda206911b9d9ea2ec371940932675066abd5365be96a2a95a`). |
| 169 | 2026-08-06 | Prepare the next separately versioned H10 + RTP specialist after Crustle as Slop Box (`teal-mask-ogerpon-ex` / Raging Bolt Ogerpon), distinct from historical Teal/Slop Box. Distill the current-deck guide from James Cox & Henry Chao heuristic acting-seat play as primary authors; use MissingNo. only as supplementary neural evidence. Steal one Cox/Chao 60-card list as the checksum-bound primary submission representative while training on the multi-deck/family system. Use the Crustle-like dual pipeline: (1) Cox/Chao(+MissingNo.) guide on the H10 RL learner as `strategic_directional_v2` training-only with zero fusion/serving/action authority; (2) separate RTP cotrain sidecar on RL shards from a frozen guide-shaped parent encoder publishing `rtp_shadow_planner.pt` / `POKEBOT_RTP_CHECKPOINT`; no guide head inside RTP serving. Guide-weighted RTP train loss requires a later explicit owner receipt. Also stage a deck-agnostic H10 warm-start core from frozen final-format Alakazam H10 `sha256:02c014ad7c3318d9871a2b16b57b25adb721d5c88cacb2a3d23db3c2f3ca0d92` and Marnie H10 training freeze `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381` via fail-closed compatibility migration and step-zero parity; recommended recipe is sequential primary-parent reuse (Marnie) then optional dual-teacher distill from Alakazam—not weight averaging—never rewriting parents. | Staged only in `state/slop_box_h10_rtp_prestage_identity_r169.json` for the post-Crustle receipt-backed boundary. Inactive: no Slop Box bootstrap/RL/RTP units started; does not interrupt Crustle iter5 restore or RTP cotrain. Exact Cox/Chao 60-card digest and owner confirmation of primary warm-start parent remain fail-closed blockers before bootstrap. |
| 170 | 2026-08-06 | Abandon the in-flight Crustle H10 specialist for now: stop/hold its managed RL/RTP/restore units via systemd only, preserve every game, shard, commit, quarantine, and receipt byte-for-byte, and do not delete or recollect. Immediately activate the separately versioned Slop Box H10 + RTP specialist (`teal-mask-ogerpon-ex`). Bootstrap trains on the full teal-mask-ogerpon-ex / Slop Box acting-seat expert corpus (not Cox/Chao-only). Distill the `strategic_directional_v2` current-deck guide from James Cox & Henry Chao (MissingNo. secondary). Steal one Cox/Chao 60-card list as the primary submission representative. Bootstrap fail-closed gate: ≥90% policy accuracy on the Cox/Chao held evaluation split, measured as acting-seat next-action argmax match (`policy_acc` / `validation_accuracy` style: `mean(predictions == target_idx)` over Cox/Chao-only held rows after `episode_id+seat+TeamNames[seat]` join). Dual pipeline: guide is RL-learner training-only with zero fusion/serving/action authority; RTP is a neural-only sidecar with no guide head in RTP serving. H10 expert bootstrap must also produce the initial Slop Box RTP cut (`rtp_shadow_planner.pt`) bound to the Slop Box H10 parent from the bootstrap trajectory/expert-derived shard—not wait for later pure-RL cotrain only. Warm-start from checksum-bound Alakazam H10 `sha256:02c014ad…` and Marnie H10 freeze `sha256:f20efb20…` via sequential primary-parent reuse (recommended: Marnie) then optional dual-teacher distill; never rewrite parents; fail-closed migration/parity before bootstrap. | Activated for abandon+start. Canonical typed pre-stage/activation identity: `state/slop_box_h10_rtp_prestage_identity_r170.json`. Crustle abandon receipt: `outputs/state/crustle-owner-abandon-r170.json`. |
| 171 | 2026-08-06 | If Slop Box Chao-hard CE overfits again—train ≫ held with Cox/Chao held stuck ≪0.90 on the same ~0.76–0.80 plateau—cross the gate under explicit owner ceiling: do not keep spinning for a measured 0.90; select the best Chao-held (fusion-valid) checkpoint; record measured fail evidence; mark ready via owner ceiling (never call 0.90 a pass); queue one nonblocking `first_if_allowed` Kaggle milestone with the Cox/Chao 60-card deck (RTP cut as sidecar when ready; do not block on remat); then register and start the Slop Box H10 self-play/public-mix RL loop with expert rehearsals every 5 iterations (Alakazam/Marnie pattern). Preserve every failed gate measurement; never label ceiling acceptance as a measured pass. Keep Jul31–Aug5 fixed-catalog remat finishing in parallel and fold added games into later rehearsals. systemd only; no Crustle deletes. | Authorized for immediate activation on clear overfit (train ≫ held, held ≪0.90) or natural Chao-hard end still short of 0.90. Receipt: `state/slop-box-owner-ceiling-overfit-proceed-r171.json`. |
| 173 | 2026-08-06 | Give Slop Box / `teal-mask-ogerpon-ex` real `combo_state` targets: versioned schema `poke_bot.slop_box_combo_state_targets/v1` mapping Teal Dance, Crispin, Glass Trumpet, Energy Switch, and engine continuity into the existing generic 32-d H10 combo head without width remap. Implement builder + attach (convert_record, pure-RL compaction, visual-trace, expert-trajectory rematerialization), keep ordinary combo loss weight `0.025`, and require nonzero labeled rows before claiming combo CE. Do not restart healthy trainers solely for metadata; no Crustle deletes. | Active immediately for label materialization and the next expert-pack / RL boundary that consumes the labeled trajectory. Receipt: `state/slop-box-combo-state-targets-r173.json` (full remat `state/slop-box-combo-state-rematerialization-r173.json`). |
| 174 | 2026-08-06 | Cap the fresh Slop Box H10+RTP expert bootstrap at outer epoch 40 (not 300). Leave healthy live CE alone until the epoch-40 checkpoint boundary; stop cleanly via systemd (no mid-epoch kill / no restart merely to change `--epochs`). Then queue/submit one nonblocking `first_if_allowed` Kaggle milestone with the Cox/Chao deck lineage from submission 55188658 (RTP sidecar if ready), then lift the RL hold only after H10+guide preflight and start self-play/public-mix RL with Crustle S-tier roster retained and matchup adapters wired from the marked corpus. | Armed for the live fresh-r171 bootstrap via `scripts/watch_slop_box_epoch40_submit_rl_r174.py` and receipt `outputs/state/slop-box-owner-epoch40-submit-rl-r174.json`. |
| 172 | 2026-08-06 | Add the abandoned non-active H10 Crustle specialist to Slop Box strong-public practice and formal premium holdout as tier `S` weight `2.0` (r111 eligible-non-active-H10 rule), alongside Alakazam H10 and Marnie H10 S-rows. Bind checksum-exact committed iter_00004 milestone package `specialist-crustle-final-format-h10-7efd8d4113e7` (checkpoint `sha256:7efd8d4113e736d28576bdbfa1c9d1c3f3a7cf1a31a0b3cfadd1e7f82cf08955`, bundle `sha256:3a380d6bd723866911d2e99e9239c679baedfdbb3ba21f27a8f2d522f7738a90`); do not use incomplete iter5 quarantine or delete any Crustle games. Expand the checksum roster from the r168 parent 19→20 opponents and formal games from 4,750→5,000 with exact 250/125/125 seats. | Bound into Slop Box runtime registry before first collect. Gate `sha256:8bcaf7e934078760bbd6c80f808c84985a1bbd8a3f0ea6ae1fe9a489ab6ca37a`, frozen `sha256:88134fcfc354b80bac2ae34e28aa31ad5ce6955f118400c9580b8e0452052da7`, package content `sha256:359e3b4fed00502e58be4631576501b6f63523226ec92f2d75446df085b19afa`. Receipt: `outputs/state/slop-box-crustle-h10-s-tier-holdout-r172.json`. |
| 175 | 2026-08-07 | Hard-swap active training to Alakazam RTP vs new fleets. Pilot the owner Abra/Kadabra/Alakazam/Dunsparce 60-card list (`decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv`; Dudunsparce promoted 2→3 to satisfy Pokémon(19)/60-card math). Loop: expert refresh on last 5 Alakazam days (2026-08-01..05) → CE rebootstrap from Alakazam `iter_00020` / `final-format-alakazam-r79-h10-refresh-v1` → Kaggle `first_if_allowed` → self-play with 1024 mirrors, fill to 8196 games, ≥1024 Grimmsnarl/set pinned uniquely to `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381` (`specialist-marnie-final-format-h10-f20efb20f5c3`). Every 5 iterations: expert refresh then Kaggle, then continue. Iteration ceiling 300. All non-combo heads live with nonzero loss (guide `strategic_directional_v2` weight 0.05; combo head explicitly off). Keep Jul24–Aug5 Slop Box CE recovery held (`ExecStart=/bin/false`). Do not resurrect failed `pokebot-final-format-alakazam-r79-h10`. | Immediate activation while no healthy conflicting trainer is running. Typed canonical source: `state/alakazam-rtp-owner-hard-swap-r175.json`. Units: `pokebot-final-format-alakazam-rtp-r175-orchestrator.service` then `pokebot-final-format-alakazam-rtp-r175-rl.service`. |
| 176 | 2026-08-07 | Build a separate localhost replay/model inspector over Elmo's downloaded Kaggle submission replay cache. For any indexed submission, game, and causally reconstructable decision step, resolve the exact immutable submitted bundle/checkpoint and display every architecture-present head value and mask, Fusion-v3 reliability and per-route/per-option contribution, policy logits/probabilities and recorded/chosen action, plus checksum-bound model parameter metadata and bounded tensor slices. Older or incomplete formats must report explicit unavailable reasons, never invent values. | Activated as the independent read-only `pokebot-replay-model-inspector.service` on `127.0.0.1:8791`. Exact submitted-runtime and per-replay byte gates are live for 126 indexed games across submissions `55315274` and `55324802`; dynamic values remain labelled `recomputed_not_historical`. Receipt: `state/replay-model-inspector-activation-r176.json` (`sha256:83ee6683cce172ed3a95647d93ba8aa4d6ab1f3138c1a81cf6736b3ec2f96d52`). It remains separate from the dashboard and training, and Kaggle/evaluation replays remain training-ineligible. |
| 177 | 2026-08-07 | Add a dashboard link that opens the separately managed Replay Model Inspector through the dashboard's existing authenticated HTTPS external-access system. Preserve Elmo's loopback-only inspector bind; carry only the fixed `/replay-inspector/` route over an encrypted, separately managed Bert-loopback-to-Elmo-loopback tunnel; allow GET only; strip browser credentials before the inspector; and do not merge inspector APIs, state, or service control into the dashboard. | Activated at `2026-08-07T23:24:53Z`. The dashboard link and authenticated HTTPS prefix are live; Bert's managed tunnel is bound only to `127.0.0.1:8792`, Elmo remains bound only to `127.0.0.1:8791`, a real 19-head/19-route trace passed through the gateway, and three independent external probes received the expected `401` challenge. Receipt: `state/replay-model-inspector-dashboard-gateway-activation-r177.json` (`sha256:f4ff6a662cb868cbb72997002f41037f709bde96f2e1d1cb9a0c7628fd0bc931`). |
| 178 | 2026-08-07 | Repair local access for LAN clients whose router cannot hairpin `mc.tsinzitari.com`: make the dashboard link same-origin and relative, and allow the direct LAN dashboard to proxy only `/replay-inspector/` over Bert's existing loopback tunnel. Keep GET-only behavior, fixed upstream and prefix stripping, private-client and cross-site rejection, browser credential/origin stripping, bounded responses, and all existing inspector/training isolation. | Activated at `2026-08-07T23:47:37Z`. The same relative link now works through the direct LAN dashboard and the unchanged external Caddy route; local root/assets/API and a real 19-head trace passed, unsafe/cross-site/non-GET requests failed closed, and ports `8791`/`8792` remain loopback-only. Receipt: `state/replay-model-inspector-lan-gateway-fix-activation-r178.json` (`sha256:da5f9fd9b16762027946ef06d87a96ddf61af73d7e03ac461773a2eeb9c956c4`). |
| 179 | 2026-08-07 | Make the Replay Model Inspector's primary decision view plain-English: transcribe each legal factorized action into what it does, keep raw action data available, and show how much each head changes the final policy using exact leave-one-head-out probability/logit effects, including the selected option and most helped/hurt legal actions. Never equate raw head magnitude with policy influence or invent values for unavailable legacy formats. | Staged for immediate implementation and read-only inspector/static-asset activation after focused causal-metric, transcript, prefix, and UI tests. Typed canonical source: `state/replay-model-inspector-human-readable-analysis-r179.json`. |
| 180 | 2026-08-07 | Display each Replay Model Inspector submission ID with its exact submitted text/label, and display both players' source-backed names, seats, and ranks for each game. Missing labels or ranks must be marked unavailable; never infer them from scores, filenames, rewards, or names. | Staged for immediate provenance/catalog and UI implementation, followed by checksum/source-binding and live replay validation. Typed canonical source: `state/replay-model-inspector-submission-player-context-r180.json`. |
| 181 | 2026-08-07 | Make Matchup Adapter use obvious for every inspected decision. Show `active for this decision`, `bypassed` with the causal reason, or `unavailable`; when active, show the matched matchup/archetype and route or slot identity, reliability, and exact policy effect if available. Installed adapter parameters alone never count as decision activation. | Staged for immediate trace-normalization and plain-English UI implementation, with active/bypass/unavailable and no-false-active tests. Typed canonical source: `state/replay-model-inspector-matchup-adapter-status-r181.json`. |
| 182 | 2026-08-07 | Split Alakazam r175 public execution by checksum-exact compatibility: use true pack-4 `LibcgMultiEnv` only for the 27 explicitly allowlisted ID+digest+group pairs; keep the 10 legacy gate packages and every unknown, changed, malformed, or cross-group package on isolated one-game remote `play`. Keep self-play pack-4, RTP, both remotes, exact per-child accounting, 32 replacement lanes, and the unchanged `1024 + 7172 = 8196` contract. Add one public-only queued request wave per remote worker without multiplying self-play queue depth. | Immediate activation at the currently stopped/quarantined iter0 boundary after checksum-aligned train/Bert/Elmo deployment, worker capability and RTP identity preflight, and focused dispatch/retention tests. Typed canonical source: `state/alakazam-public-multi-env-split-r182.json`. |
| 183 | 2026-08-07 | Make the direct local Replay Model Inspector route work through Bert's Tailscale hostname/address. Treat only the actual socket peer in `100.64.0.0/10` as an allowed private-overlay client in addition to loopback/RFC-private/link-local peers; keep public and adjacent ranges rejected and retain every existing Host/Origin, GET-only, path, fixed-upstream, credential-stripping, size, tunnel, and service-isolation boundary. | Staged for immediate dashboard-only activation after boundary tests and live `bert:8780` verification. Typed canonical source: `state/replay-model-inspector-tailscale-local-gateway-r183.json`. |
| 184 | 2026-08-08 | Add Kaggle submission `55217604` to the replay index as a permanent explicit special case. Keep the hourly timer, minimum discovery ID, automatic new-owner-submission discovery, and explicit CLI override behavior unchanged; union the special ID only into the default discovered set and recheck it hourly for new games. Never borrow r175 model provenance when its own artifacts are absent. | Active. The unchanged hourly timer rechecks the special ID while preserving the `55315274` discovery floor and ordinary new-submission discovery. All 79 cached games and the exact checkpoint weights are indexed; dynamic traces fail closed because this submission's runtime package/parity identity is not yet attested, and r175 is never substituted. Receipt: `state/replay-model-inspector-submission-55217604-special-case-activation-r184.json`. |
| 185 | 2026-08-08 | Matchup Adapters must be on for future Kaggle submissions with a trained bank. Require the verified runtime-enabled tree and exact submitted startup activation at packaging; make the inspector reproduce that checksum-bound startup state request-locally instead of treating the serialized dormant training flag as serving truth. Active requires an accepted routable route; unknown/unroutable remains an explicit bypass. | Inspector correction active: exact submitted startup is reproduced request-locally, cache state is restored, and a live Alakazam route-6 decision reports the adapter active with exact policy influence. The stricter future-package build gate is staged for the next submission build boundary and has not rewritten historical packages. Receipt: `state/replay-model-inspector-submitted-adapter-runtime-activation-r185.json`. |
| 186 | 2026-08-08 | Keep the exact numeric submission ID visible in every inspector submission option and selection summary, and add the source-backed cached-replay win rate with explicit wins/eligible-games denominator. Exclude and report missing/malformed acting-seat outcomes rather than guessing. | Active read-only. Live cached outcomes show `48/79`, `55/78`, and `40/58` for submissions `55217604`, `55315274`, and `55324802`; missing/malformed outcomes remain excluded. Receipt: `state/replay-model-inspector-submission-win-rate-activation-r186.json`. |
| 187 | 2026-08-08 | Add two clearly separated playground views: (1) instant Decision Influence scales (`0x`–`2x`, `1x` exact baseline) that recompute the actual nonlinear fusion path for this decision and show policy/action changes; (2) source-backed Training Weight Recipe values such as guide `0.05`, which are learning-loss multipliers and cannot change a forward pass without later fine-tuning. Never conflate the two or persist changes to model/training/runtime state. | Active read-only. Live `0x`, `1x`, and `2x` requests use exact nonlinear recomputation without checkpoint mutation; `1x` matches the runtime baseline and `0x` matches exact leave-one-head-out. The separate checkpoint-bound recipe displays guide loss `0.05` and has no fine-tune authority. Receipt: `state/replay-model-inspector-head-influence-playground-activation-r187.json`. |
| 188 | 2026-08-08 | Show only Challengestone's own submitted-agent decisions in the inspector selector and decision-step API. Resolve the acting side from each replay's archived own-agent seat rather than assuming a fixed seat; keep opponent events only as hidden causal history and reject direct opponent-step traces. | Active read-only. Live games with Challengestone in seat 0 and seat 1 expose only the submission-bound own decisions; direct opponent addresses return `opponent_decision_not_selectable`, while opponent events remain hidden causal history. Receipt: `state/replay-model-inspector-own-seat-decisions-activation-r188.json`. |
| 189 | 2026-08-08 | Render every user-facing submission ID as an exact base-10 string such as `55315274`; never pass identifiers through metric formatting that can produce scientific notation, rounding, grouping, or abbreviation. | Active read-only across selector, summaries, game/trace context, parameter inventory, and provenance panels. Live browser validation found and repaired the final technical-panel numeric formatter path; no scientific notation remains. Receipt: `state/replay-model-inspector-submission-id-text-activation-r189.json`. |
| 190 | 2026-08-08 | Add a mobile-friendly game search field that filters the current submission's game dropdown by full or partial exact-decimal game ID or available player name, while retaining the native select as the final chooser. | Active read-only. Mobile browser validation passed partial ID (`60007`), player (`kura`), no-match, clear/reset, count, and first-match selection behavior over the live 79-game special-case list. Receipt: `state/replay-model-inspector-game-filter-activation-r190.json`. |
| 191 | 2026-08-08 | Add an explicit Matchup Adapter ON versus OFF view for every trace-ready decision where the checksum-bound submitted runtime actually applied a routable adapter route. Show both choices, per-action probabilities, and signed changes from an exact no-adapter forward pass; never force an unknown route or use an unattested runtime. | Active read-only. A live route-6 Alakazam decision shows the submitted-runtime ON choice beside the exact no-adapter OFF rerun across all 13 legal actions, with signed ON−OFF probability changes; the mobile table scrolls horizontally and cached model state is restored. Receipt: `state/replay-model-inspector-matchup-adapter-counterfactual-activation-r191.json`. |
| 192 | 2026-08-08 | Add exact H10 Marnie `specialist-marnie-final-format-h10-f20efb20f5c3` / checkpoint `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381` as a distinct additional Alakazam r175 specialist at tier `S++`. Keep it separate from historical old-format Marnie; define scoped `S++` as weight `4.0`, retain the existing ≥1,024-game/set H10 Marnie floor, and do not change the exact 8,196-game total. | Activated from the exact iteration-16 inactive-boundary receipt after commit `sha256:d02c56cf59294236121998399965c24cba111b2e44af9522b58dea3ce0ef114f`; no iteration-17 artifact existed before activation. The one-shot migration passed, and iteration 17 sealed plan `sha256:94e39ed349f8a0384f72a5a58e6f80f314101d951c5e7141a26c3fd8e5393653` with 18 distinct gate rows, H10 Marnie `1,049` games against its `1,024` floor, and unchanged `8,196` total. Activation receipt `sha256:6a2ccf609c0ab3d2b4fc618f432b7fda8eeabc5cbd5426350b0130860b1e1497`; post-activation runtime-parity repair receipt `sha256:8e95a020584cb48526ab67b39441a691f0e74cf24647a671c620292b22435817` proves the candidate's r193/PokeRLM/RTP package parity and real iteration-17 RTP pack-4 dispatch. Typed source `state/alakazam-marnie-splusplus-opponent-r192.json`. |
| 193 | 2026-08-08 | Run one large Alakazam r175 expert refresh, then resubmit its exact refreshed checkpoint to Kaggle. “Large” is exactly 25 full-model expert epochs (5× the ordinary rehearsal) over the checksum-pinned 2026-08-01..05 Alakazam corpus; keep guide `0.05`, setup `0.025`, combo loss/route off, and all other architecture-present learner heads trainable. | Activated after durable iteration 14 and completed. The immutable refresh receipt records 25/25 epochs and expert checkpoint `sha256:819759347faf…fcd0`; iteration 15 retained exactly 8,196 source games and committed checkpoint `sha256:c2b01f5a12a4…f69c`. The exact `first_if_allowed` bundle is `sha256:c8cbad271829…b0f2`; Kaggle submission `55359777` is `COMPLETE` / accepted and currently reports score `926.6`. Ordinary cadence resumed into iteration 16. Activation and completion evidence is checksum-bound in `state/alakazam-large-expert-refresh-resubmit-r193.json`. |
| 194 | 2026-08-08 | Submit one additional Kaggle copy of exact Alakazam iteration-15 checkpoint `sha256:c2b01f5a12a4164e282f278e104da3dd5d5b0c1467d592e01d832be141fcf69c`. Reuse the checksum-verified revision-193 bundle bytes and r175 pilot deck, retain `first_if_allowed`, and bind the new copy to its own unique label, queue identity, one-shot authorization, and receipt. | Completed without interrupting training. The checksum-identical copy-2 bundle `sha256:c8cbad271829…b0f2` was accepted as Kaggle submission `55362452`, status `COMPLETE`, score `600.0`. Submission `55359777` and all model/runtime bytes remain unchanged. Typed source: `state/alakazam-iter15-second-copy-r194.json`. |
| 195 | 2026-08-09 | Run one additional 25-epoch full-model Alakazam expert bootstrap from immutable terminal iteration 20, then submit two new first-preferring r175-pilot-deck bundles from that same checkpoint: copy 1 with RTP fully disabled and copy 2 with the checksum-bound canonical Alakazam r175 RTP sidecar enabled. Preserve all non-combo heads, guide `0.05`, setup `0.025`, combo loss/route off, and revision-185 Matchup Adapter runtime in both. The visible messages must include exact text `NO RTP` and `RTP`; package/startup proof must independently prove each runtime state. | Completed. The 25/25 checkpoint is `sha256:261d367e…9cc3a`. Kaggle submissions `55378392` (`NO RTP`, bundle `dfa8bfcc…`, score `500.4`) and `55378477` (`RTP`, bundle `2f982f25…`, score `600.0`) are both `COMPLETE`; the RTP bundle binds sidecar `sha256:dde7b813…`. Typed source: `state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json`. |
| 196 | 2026-08-09 | Save the exact immutable submitted bundle, packaged runtime, checkpoint/model, matchup tree, label, and submission-ID binding on Elmo for every accepted owner submission. Backfill the full currently indexed range and archive future accepted queue rows automatically; reconstruction remains fail-closed until exact runtime parity is attested. | Backfill complete for the current index: 18/18 submissions are checksum-verified and trace-ready, with 1,102/1,102 cached replay games available. The recurring archive remains hourly for future accepts. Typed source: `state/replay-model-inspector-all-submission-artifact-archive-r196.json`. |
| 197 | 2026-08-09 | Realign Alakazam RTP on production as a new checksum-bound candidate tied to the exact r195 parent and protected Aug-1--5 corpus. Use complete ordered serving actions, outcome/value-of-planning supervision, whole-game heldout evidence, strict parent/config/promotion binding, and a three-arm NO-RTP/direct-bridge/recursive-RTP evaluation. Set the initial exact hard neural-pass ceiling to 32 while proving the current recursive path requires and completes within 6; separately measured and checksum-bound profiles may rise only as needed, with an absolute owner ceiling of 256 and no automatic escalation. | Authorized for immediate implementation and production shadow training at the inactive terminal boundary. Preserve r175/r195 and the old sidecar; do not restart r175, collect iter21, change the selector, grant action authority, or submit to Kaggle until the separately recorded gates and promotion receipt authorize those later actions. Typed source: `state/alakazam-rtp-realignment-r197.json`. |
| 198 | 2026-08-09 | Before r197 materialization, set the complete ordered legal-action cap to exactly 1,024 and the current `pure_rl_r197` neural-pass ceiling to exactly 256. The measured skeleton still needs only 6 normal / 5 forced-replan passes; 256 is a hard ceiling with no automatic escalation above it. | Production shadow materialization completed successfully from source snapshot `2ae56bc6a2db…` as candidate `bc31f860b815…`; exact 1,024/256 and observed 6/5 were reverified. The candidate remains shadow-only pending the required three-arm matched evaluation and separate promotion receipt; r175/r195, selector, serving, and Kaggle restrictions remain intact. Typed source remains `state/alakazam-rtp-realignment-r197.json`. |
| 199 | 2026-08-10 | Continue iterating on Alakazam RTP as separately versioned, receipt-backed, shadow-only R&D; rescind the provisional abandonment instruction before it activates. Treat r198 attempt 10's poor live prefix as non-terminal telemetry rather than an efficacy, promotion, abandonment, or stop decision. | Preserve and finish attempt 10 unchanged through immutable terminal evidence, never retry or rewrite it in place, and preserve every HOLD, rejection, invalid attempt, or failed gate. Any follow-up requires a new content-addressed source, candidate, evaluation identity, and output root after receipt-bound diagnosis and preflight. No gate is weakened and serving, action, selector, checkpoint-publication, submission, promotion, r175 restart, iteration-21, and Kaggle authority remain false. Typed source: `state/alakazam-rtp-continuation-r199.json`. |
| 200 | 2026-08-10 | Work on improving RTP or a different GPU turn-planning strategy. Pursue a separately versioned conservative batched one-turn complete-action GPU reranker instead of extending the current stale multi-action recursive program path. | Research implementation is authorized offline and shadow-only while attempt 10 continues untouched. Base-policy parity is mandatory unless trusted paired counterfactual targets plus calibration, uncertainty, support, and margin gates permit an override. No r197 target fabrication or hidden-information/evaluation/Kaggle leakage is allowed. Runtime attachment requires new content-addressed source, sidecar, candidate, evaluation, output, latency/reliability, and promotion evidence; every production authority remains false. Typed source: `state/alakazam-gpu-turn-planner-r200.json`. |
| 201 | 2026-08-10 | Clarify that “one turn” means planning multiple atomic actions through the end of the current turn, not one-step action reranking. Supersede revision 200 before implementation or activation. | Build a separately versioned closed-loop receding-horizon full-turn GPU planner: plan a bounded current-turn trajectory, execute one action, observe the real result, then replan the remainder. Never cross turns, blindly resolve branches, or reuse stale state/legal encodings. The direct action stays a mandatory fallback and no override occurs without trusted multi-step dynamics/counterfactual/calibration/support/margin/legality/latency/reliability evidence. Attempt 10 and all authorities remain untouched. Typed source: `state/alakazam-closed-loop-turn-planner-r201.json`. |
| 202 | 2026-08-10 | Replace unconditional post-action tree rebuilding with chance-aware cached inter-turn planning. Reuse an exact matching deterministic subtree without recalculation; expand a simple fully enumerated chance point with exact probability-weighted value, but stop at hidden, incomplete, complex, or unbounded chance/information boundaries. Use one easy-to-change typed budget object with defaults of 20 seconds per actual turn and 5 seconds before an atomic action. | Authorized for isolated offline phase-1 state-machine implementation only. One real action is followed by exact state/legal/encoding validation before child advance; missing outcomes or fingerprints never default. Effective budget values are config-identity-bound and timeouts return the exact direct action. MCTS/expectimax runtime claims, services, selector, serving/action authority, publication, submission, promotion, r175 restart, and iteration 21 remain forbidden pending all prerequisite receipts. Attempt 10 remains unchanged. Typed source: `state/alakazam-chance-aware-inter-turn-mcts-r202.json`. |
| 203 | 2026-08-10 | Make every indexed owner submission trace-ready; expose deterministic setup-runtime truth plus a clearly separate hypothetical archived-model rerun; run forward passes on Elmo's GPU; make submissions searchable and steps directly addressable; and warm every step/stage of a selected game into a device-local cache. | Active read-only. All 18 indexed submissions and 1,102 replay games are verified. Submission `55378392`, game `91468417`, step `15` returns six GPU policy probabilities and 19 heads; its cold load measured 17.074 seconds and its warm response 0.597 seconds. Step `1` separately shows actual runtime 100%/0% and the hypothetical archived-model softmax. Typed activation source: `state/replay-model-inspector-deterministic-runtime-policy-r203.json`. |
| 204 | 2026-08-10 | Explain every policy head for nontechnical readers, including what it is looking for, its time horizon, and naming traps; show its exact current-decision policy effect; add the current-deck guide as a production shadow-only second opinion; and put newest submissions and game/episodes at the top while keeping steps chronological from the beginning. | Active read-only inspector extension. The 19-head FAQ distinguishes eventual loss, near-term knockout danger, and our offensively named `lethal_threat`; current influence is exact leave-one-head-out rather than a fixed or additive percent weight. The checksum-bound guide shadow reports its recommendation and agreement while proving zero logit/action authority. Typed source: `state/replay-model-inspector-head-faq-guide-shadow-r204.json`. |
| 205 | 2026-08-10 | Run exactly 1,000 same-checkpoint Alakazam mirror games: 500 RNG-matched seat-swapped pairs comparing the real chance-aware inter-turn MCTS arm with the frozen revision-195 NO-RTP direct policy. Enforce 20 seconds per actual turn and 5 seconds per atomic action, use safe compatible idle remotes in parallel, perform no additional training, and report outcomes plus per-turn result/leaf counts and full-tree-within-budget completion. | Authorized for implementation, preflight, and the exact shadow BO1000 only after every real-search, successor/future-legality, exact-chance, hard-clock, integrity, parity, determinism, safe-remote, and content-addressed-output prerequisite passes. The phase-1 cache validator alone is not MCTS. Attempt 10 remains untouched; evaluation data is training-ineligible and all production authority remains false. Typed source: `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json`. |
| 206 | 2026-08-10 | Submit two NO-RTP terminal Alakazam guide-logit A/B variants from the exact revision-195 bundle: normalized bounded guide bonus `0.05` versus `0.10`, with exact model fallback whenever the guide is unavailable or non-unique. | Authorized for immediate immutable packaging and the existing receipt-backed Kaggle queue. Both copies retain the exact checkpoint/deck/matchup/search identities, carry explicit NO-RTP and guide-weight labels, and do not mutate or replace historical submissions. Typed source: `state/alakazam-guide-logit-ab-submissions-r206.json`. |
| 207 | 2026-08-10 | Supersede only r205's experimental-arm mechanics: use simulator-backed chance-aware inter-turn MCTS with checksum-bound frozen-model policy priors, batched frozen outcome/value reranking for nonterminal leaves, and exact simulator terminal results. Preserve the exact BO1000/r195 pairing, 20s/5s hard clocks, no-training boundary, and split telemetry. | Staged for offline implementation/preflight and the exact shadow BO1000 only after simulator, exact-terminal, frozen-model, clock, integrity, determinism, host-safe-noninterference, and content-addressed-output receipts. Bert, Elmo, and train remain unavailable until per-host noninterference passes; all production authority remains false. Typed source: `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json`. |
| 208 | 2026-08-10 | Show each Replay Model Inspector playground head's actual nominal baseline Fusion coefficient, separately from its `1.0x` source multiplier. | Active read-only inspector extension. The live selected-decision API displays `cap * learned multiplier / active-route count` and its components while labeling it pre-nonlinearity, not a fixed percent contribution; exact recomputation remains policy-effect truth. No A/B analysis, training, submission, or model mutation. Typed source: `state/replay-model-inspector-baseline-head-coefficients-r208.json`. |
| 209 | 2026-08-10 | Add an on-demand `Check Kaggle now` button so newly completed submissions and episodes can be pulled between the unchanged hourly refreshes. | Active. One authenticated/private, custom-header, bodyless POST invokes only the existing fixed Elmo replay-sync oneshot; the UI polls its status and refreshes the index when it finishes. Live checks returned 202 for the exact request and 405 for missing-intent or other POST routes. The hourly timer remains enabled. Typed source: `state/replay-model-inspector-manual-replay-sync-r209.json`. |
| 210 | 2026-08-10 | Fully abandon legacy recursive RTP and immediately stop the active r198 attempt-10 evaluation. Preserve the incomplete prefix and every historical r197/r198/r199 artifact; never retry, restart, train, evaluate, attach, serve, promote, publish, select, or submit that RTP line. | Activated immediately through the managed systemd unit. The stopped prefix has 761 immutable transcripts/receipts, 253 complete matched cells plus two arms, zero failed-worker evidence, and no terminal result/compiler/HOLD/promotion artifact; it is not a complete efficacy result. A persistent external systemd drop-in now refuses manual starts and skips indirect activation before evaluator code while preserving the exact linked unit; receipt `state/alakazam-rtp-abandonment-retirement-guard-r210.json` (`sha256:d0ee2255bf2b5e4abd2c1b9eaaff39343997c2452578d53347927fa5b2f75db0`). The separately versioned r207 simulator-backed MCTS work is explicitly non-RTP, may not consume the abandoned RTP sidecar/executor, and remains shadow-only pending every prerequisite. Typed source: `state/alakazam-rtp-abandonment-r210.json`. |
| 211 | 2026-08-10 | Make newly archived submissions trace-ready automatically and reproduce any exact package-local guide decision layer rather than stopping at the neural checkpoint. | Active. Submissions `55410353` and `55410425` are trace-ready with their exact `0.05` and `0.10` package policies. The live trace preserves neural-only probabilities and shows final guide-adjusted submitted-runtime probabilities/action; exact fallback remains visible when the guide evidence is tied or unavailable. Future verified archives receive submission-specific runtime attestation during the same managed provenance refresh. This remains recomputed, not historical, and grants no A/B analysis or mutation authority. Typed source: `state/replay-model-inspector-new-submission-reproduction-r211.json`. |
| 212 | 2026-08-10 | Train one isolated 100k--500k parameter Alakazam Guide2Vec head from the frozen r195 NO-RTP policy representation and compact causal guide targets, then run exactly 1,000 no-MCTS/no-RTP mirror games versus the identical direct policy. Require 500 matched pairs with explicit actual first/second balance, separate Blackwell managed isolation, and no Slop Box or evaluation-data training. | Authorized to start only after the dedicated Blackwell noninterference/materialization/frozen-identity receipts pass. The guide head alone may receive gradients; both the base model and all production, selector, serving, promotion, Kaggle, r175-restart, and iteration-21 authority remain false. Typed source: `state/alakazam-guide2vec-no-mcts-bo1000-r212.json`. |
| 213 | 2026-08-10 | Add a presentation-only PTCG Visualizer link for each selected archived Replay Model Inspector game, using only its exact decimal replay ID and opening without referrer or payload forwarding. | Active read-only inspector change. It has no external write, training, submission, or replay-mutation authority. Typed source: `state/replay-model-inspector-ptcg-visualizer-link-r213.json`. |
| 214 | 2026-08-10 | Run a separate testing-only 1,000-game direct-r195-NO-RTP versus simple public-history root-sampled BeliefMCTS BO1000. Both arms use the same frozen checkpoint, bundle, deck, full model, and trained Matchup Adapter runtime/tree; only action selection differs. Disable RTP, legacy RTP, guide-linear/guide-logit, and Guide2Vec in both arms. Use 500 seeded RNG/deck-order matched seat-swapped pairs, exact 500/500 MCTS seat and actual-first/second balance, and typed hard 20s-per-turn / 5s-per-action monotonic budgets. | Authorized for implementation, preflight, and the exact shadow BO1000 only after frozen-package/adapter parity, real-search, absence, timing, pairing, determinism, noninterference, and content-addressed-output receipts. This root-sampled hidden-particle/coin-sampling experiment is explicitly non-r207 exact-chance; r207 remains preserved. No training, serving, selector, publication, Kaggle, promotion, r175 restart, or iteration-21 authority. Typed source: `state/alakazam-simple-belief-mcts-bo1000-r214.json`. |
| 215 | 2026-08-10 | Supersede r214's unlaunched per-decision search semantics with a separately versioned full-actual-turn root-sampled BeliefMCTS BO1000. Preserve r214. Search at the first atomic decision of an actual turn, then continue only a fingerprint-validated deterministic cached branch within that same turn; clear it at a new turn and rebuild only on divergence or chance/information boundaries with the remaining shared turn pool. Within the turn tree, evaluate each unique deterministic model-input state once, reuse frozen policy/value thereafter, and prove real simulator successor expansion, terminal handling, value backups, and multi-step depth. A private hypothetical simulator action may never reach the real game. Public observation equality never merges a transposition: require a native exact semantic-state attestation (hidden/RNG/pending effects/selection/config/future legal order) and an optional valid `ActionsCommute` certificate before skipping a second order; unavailable proof expands separately. Replace the obsolete fixed 50-sim target with time-bounded search: source-backed 600s outer game clock allocates a dynamically shrinking pool capped by the easy-to-change 20s per-turn default; 5s is a per-operation ceiling and every effective allowance is bounded by remaining turn and game time; minimum one valid sim, direct fallback if none, and only a very high emergency safety ceiling. | Authorized for implementation, preflight, and the exact shadow BO1000 only after package/adapter parity, real full-turn search/cache/chance/timing/pairing/determinism/noninterference/content-addressed-output receipts. This remains root-sampled non-r207 exact chance, training-ineligible and shadow-only. No training, serving, selector, publication, Kaggle, promotion, r175 restart, or iteration-21 authority. Typed source: `state/alakazam-full-turn-belief-mcts-bo1000-r215.json`. |
| 216 | 2026-08-10 | Authorize a local exploratory BO1000 using the existing approximate BeliefMCTS/search APIs so the run can start without waiting for perfect native semantic-state equivalence or `ActionsCommute` proof. Preserve r215 byte-for-byte for exact/promotion-qualified work. Both arms remain the same frozen r195 NO-RTP model/package/deck and runtime-on Matchup Adapter; RTP and every guide runtime layer remain off. The 600s clock uses `min(20.0, max(0.0, (remaining_game_seconds - 30.0) / 8.0))`, so healthy games receive the full 20s and only shrink below 190s; the 5s component cap is inside that shared residual turn pool. Run 500 seeded seat-swapped pairs / 1,000 games with exact seat and actual-first/second balance. | Local-only, training-ineligible, non-exact, and non-promotion. Results must carry non-exact/non-r207/non-promotion labels and may not claim native exact-state, commutation, or transposition savings. No Kaggle API, queue, upload, or submission; no training, serving, selector, publication, promotion, r175 restart, or iteration-21 authority. Typed source: `state/alakazam-local-approximate-belief-mcts-bo1000-r216.json`. |
| 217 | 2026-08-10 | Clarify r212 before launch: keep the exact frozen r195 Matchup Adapter/tree ON for latent extraction and both mirror arms; candidate has exactly one frozen Guide2Vec component, while control has no Guide2Vec module, parameter, state key, hook, linear transform, disabled component, or zeroed component. Historical guide-linear/guide-logit layers remain absent. | Canonical immediately for the still-unlaunched r212 path. Require graph-absence/difference and adapter-parity receipts; do not alter any other BO1000. Training/evaluation remain isolated and shadow-only with no production, selector, promotion, or Kaggle authority. Typed source: `state/alakazam-guide2vec-no-mcts-bo1000-r212.json`. |
| 218 | 2026-08-10 | For a new local approximate BO1000 only, preserve r216 byte-for-byte but search solely at the first actual decision of each turn for up to `min(10s, dynamic game allowance)`. The whole first-decision search-or-fallback operation is capped at that allowance; at full allowance it may reserve 9.5s for private search and 0.5s for direct fallback, while individual calls are telemetrized without an inherited 5s outer call cap. Later same-turn decisions must use a fingerprint-validated cached plan or frozen direct fallback and may never launch fresh search. Eliminate fixed simulation/depth targets; retain an emergency guard only, and permit early search stop only after explicit stable-root convergence plus a fully backed-up legal action. A future separately authorized Kaggle runtime should target AWS p5.4xlarge-equivalent H100 80GB / 256 GiB / 16 vCPU with batched inference and resource-aware search. | Staged design/projection only. Fresh r218 preflight and content-addressed output are required before any execution; this record launches or changes no runtime, service, RTP path, selector, training, or Kaggle operation. Typed source: `state/alakazam-local-first-decision-belief-mcts-bo1000-r218.json`. |

| 219 | 2026-08-10 | Correct r218 for a new local approximate BO1000 only: one dynamically bounded 45s planner pool per actual turn, with every meaningful search segment—including the first—capped at 15s and later meaningful boundaries able to spend only the residual pool. Deterministic cached/obvious/forced steps validate and dispatch without forced search; actual turn end closes its pool/cache. Fully exposed finite chance points of at most six outcomes use complete exact probability-weighted backup and can continue through their children within the current segment/turn budget; opaque chance/info re-roots after reality. PUCT prioritizes high frozen priors but does not threshold-prune positive legal lines. No fixed sim/depth target exists beyond an emergency guard, and convergence remains backed-up/legal only. Run a 10-game/5-pair valid canary reporting multi-search telemetry before the 500-pair/1,000-game mirror. | r218 remains byte-for-byte history. Fresh r219 preflight and a valid managed canary are required before BO1000; then local execution is authorized. Both arms remain frozen r195 NO-RTP with Matchup Adapter on and all RTP/guide runtime layers off. Training, serving, selector, promotion, checkpoint publication, Kaggle, and legacy-RTP authority remain false. Typed source: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json`. |
| 220 | 2026-08-10 | Let Replay Model Inspector quick find accept a pasted replay URL and use its decimal `submissionId` and `episodeId` query parameters to open that exact indexed replay. Ignore unrelated parameters; never fetch or navigate to the pasted URL; never fall back to another episode when the requested one is absent. | Activated on the live read-only gateway at `2026-08-10T19:34:45Z`. The served HTML and exact tested JS/CSS hashes match, submission `55410353` / episode `91735935` resolves through the live API to 107 decision steps, and the inspector remained healthy on the same PID with no restart. Automated browser interaction remains pending because no controllable browser was connected; owner refresh confirmation is the remaining visual check. No replay sync, training, selector, or submission authority was added. Typed source: `state/replay-model-inspector-replay-link-quick-find-r220.json`; activation receipt: `state/replay-model-inspector-replay-link-quick-find-activation-r220.json`. |
| 221 | 2026-08-10 | Supersede only r219's stochastic fallback for a fresh local multi-search-turn MCTS mirror. Exact fully forceable finite chance of at most six outcomes remains allowed with exact probabilities, independent successors, future legality, same-pre-random-state forcing, and probability-weighted backup. Paired-engine seeding is only for match reproducibility; it may not hunt or pre-randomize desired chance outcomes. Every unforceable or incompletely proven random event instead stops at the pre-random boundary for frozen leaf evaluation: never privately sample a coin/die/outcome, guess game rules/distributions/successors/future legality, or advance an unobserved outcome. After reality, a later meaningful decision may re-search only from the residual 45s shared turn pool. | Staged for a new content-addressed r221 preflight and 10-game/5-pair canary before BO1000. r219 remains byte-for-byte preserved; its 45s pool, 15s segments, multi-search, deterministic cache, frozen r195/Matchup Adapter parity, full 500-pair BO1000, and no-training/no-serving/no-Kaggle boundaries remain binding. Typed source: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r221.json`. |
| 222-MCTS | 2026-08-10 | Supersede only the r221 local-MCTS separate-canary sequencing and evaluation transport. Launch one fresh-preflight Blackwell 500-seat-swapped-pair/1,000-game evaluation; pairs 0–4 / games 0–9 are an in-job diagnostic only, never a pause/restart/authorization gate. Use exact stock r195 `cg/libcg.so` in a fresh OS process per game, in-process stock Search for private MCTS, and shared Blackwell queued GPU leaf inference—never B77, seeded/`BattleStartSeeded`, batch/multi-game custom engine, or custom chance force path. Each decision segment has exactly eight concurrent trajectories selecting/reserving from one shared logical tree, isolated stock Search states, and microbatched leaf forwards/backups into that same tree; no independent root forest/merge, serial fallback, or partial-lane MCTS action authority. Virtual-loss/path/leaf reservations and safe in-flight frozen-eval coalescing/cache avoid duplicate work, without public-lookalike hidden/random-world merges; zero reservations remain at action return. This is not eight games/models or literal beam search, and lack of eight-state isolation hard-fails preflight. Pair RNG streams are independent/unmatched and must not be called paired RNG. Preserve r221 pre-random boundary semantics unless the stock ABI itself proves exact forceability. | Staged for fresh r222 preflight and one managed Blackwell launch. Report requested/active lanes, isolation, shared-tree, leaf-microbatch, dedup/cache/unavoidable-repeat, and zero-reservation telemetry without imputation. A separate stock portable-Kaggle compatibility smoke is required but nonblocking for the local BO1000 and authorizes no Kaggle action. r221 remains byte-for-byte preserved; its 45s/15s search, frozen r195/Adapter parity, no-training/no-serving/no-selector/no-Kaggle constraints remain binding. Typed source: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json`. |
| 222 | 2026-08-10 | Make Replay Model Inspector reconstruction selected-trace-only on Elmo's RTX 3060 and bound browser-visible selected-trace wait to 20 seconds. Remove whole-game background prefetch, abort stale browser requests, cache only traces actually requested, and make submission `55410353`'s exact package/checkpoint the one resident service runtime. | Implemented and staged for an inspector-only managed restart after static, unit, exact-runtime, health, and cold/warm ≤20s validation. Preserve exact causal/checksum gates and `recomputed_not_historical` labels; other runtime packages remain fail-closed isolated. No replay, training, selector, submission, or checkpoint authority is added. Typed source: `state/replay-model-inspector-on-demand-gpu-trace-r222.json`. |
| 223 | 2026-08-10 | Make all five headers in the Matchup Adapter ON/OFF legal-action comparison sortable: Legal action, Adapter ON chance, Adapter OFF chance, signed Change caused by adapter, and Chosen. | Implemented as a client-only, stable, accessible presentation change and staged with r222's static activation. Default source order and exact source values remain unchanged until a header is clicked; no API, replay, model, training, or selector semantics change. Typed source: `state/replay-model-inspector-adapter-comparison-sorting-r223.json`. |
| 224 | 2026-08-10 | Stage a Phase-1 ladder submission candidate that attempts exactly eight simultaneous root-parallel belief-search lanes. Each persistent lane owns a distinct raw `AgentStart()` handle and dedicated thread; never share the stock `cg.api.agent_ptr`. Share only frozen-model inference through one queue-owned micro-batching broker, deterministically merge all eight complete canonical root visit vectors, and use the exact frozen direct policy if any lane, deadline, isolation, completeness, or integrity check fails. | Staged only; no current package, selector, service, queue, upload, or Kaggle submission is changed or authorized. Activation requires crash-contained native-handle isolation, parity, memory, throughput, deadline-cleanup, and actual Phase-1 resource receipts plus separate submission authorization. The candidate requests and selects eight lanes with no automatic lane reduction; published Phase-1 assumptions remain 2 vCPU, 12.2 GiB RAM, and no GPU. Typed source: `state/alakazam-phase1-ladder-eight-lane-search-r224.json`. |
| 225-MCTS | 2026-08-10 | After all local exact-package, shared-tree eight-lane isolation/cleanup/randomness/throughput preflights pass, conditionally authorize exactly one `pokemon-tcg-ai-battle` diagnostic labeled `DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY`. It must hard-fail without eight active isolated stock Search states selecting/reserving/backing up into one shared logical tree, frozen-model leaf microbatching, and zero outstanding reservations. | A separately bound one-shot only: immutable receipt must bind package, members, entrypoint, r222/r225, stock libcg, frozen r195 assets, competition, label, and local preflight. No retry/copy/queue, review/strength/selector/promotion/gameplay authority; ordinary r222 BO1000 stays local/continuous. Expected diagnostic resource receipt is p5.4xlarge-equivalent H100 80GB/256GiB/16vCPU. Preserve distinct r224 root-parallel 2-vCPU/no-GPU contract byte-for-byte. Typed source: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 226-GUIDE2VEC | 2026-08-10 | Abandon the unlaunched r212 Alakazam Guide2Vec training/BO1000 while preserving its artifacts, and stage a dormant guide-agnostic numeric Guide2Vec pipeline. Separate reusable frozen base latents from guide-versioned label overlays keyed by exact causal stage and legal-option fingerprints so a future guide update relabels and retrains only the tiny head. | Pipeline implementation and read-only checks only. No guide is currently ready; r212 launch authority, training, gradients, candidate publication, runtime attachment, BO1000, selector, serving, promotion, Kaggle, RTP, and MCTS changes remain unauthorized. A later explicit owner launch plus sealed guide/alignment/split/base/host/source/output receipts is required. Typed source: `state/guide2vec-general-training-pipeline-r226.json`. |
| 227 | 2026-08-10 | Correct only r222/r225 topology: one Kaggle submission process and one loaded stock libcg DSO host eight internal `AgentStart` simulator/search arenas, one per persistent CPU worker, not eight competition agents. Each arena has one distinct `SearchBegin` ID; a master owns one tree, gathers eight frontier leaves into one frozen GPU batch, backs up all eight, then repeats. | Staged documentation/canonical correction only; no runtime, service, or Kaggle action. No one-lane baseline/ratio, per-lane GPU batch, private-tree/root merge, reduced lane, or partial-lane authority is allowed. All other r222/r225 semantics remain unchanged. Typed sources: `state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r222.json`, `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 228-MCTS | 2026-08-10 | Make the rough one-shot Kaggle full-gameplay viability test the active r225 deliverable ahead of BO1000. At every branching gameplay decision, one submission process and one loaded stock r195 DSO host eight persistent internal `AgentStart` simulator arenas / CPU lanes searching one master-owned shared MCTS tree: `SearchBegin` exactly once per lane/decision, exact `(lane, handle, SearchId)` retained across depth waves, simulator advances to discover next legal actions, up-to-eight returned frontier states go through frozen r195 GPU evaluation, and all results back up into the sole tree. | Documentation/typed-contract staging only; r222 stays byte-for-byte unchanged. Require a local exact-package full-game smoke before exactly one direct `DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY` submission, with no queue/retry/copy. A clean deadline after full cleanup returns the best legal fully backed root action, or, when it alone caused zero backups, the frozen direct fallback. Any non-deadline zero backup or structural/integrity failure exits nonzero; only forced single-action prompts bypass search. Emit one explicit success marker only after the full gameplay loop passes. |
| 236-LIBCG | 2026-08-10 | Make the official Kaggle Environments 1.32.6 CABT native-library set the canonical libcg for every new r235 Kaggle and r229 BO1000 package. Bind all four platform binaries and forbid old/new mixing; preserve every old package/result under its actual simulator identity. | Staged immediately for offline source/package/preflight updates. r229 retains the r233 outer game watchdog and excludes the r234 Kaggle broker/cleanup lifecycle; revision 236 changes only native simulator identity there. No managed-service restart, training, or extra Kaggle authority. Typed source: `state/canonical-libcg-r236.json`. |
| 238-KAGGLE | 2026-08-10 | For the one pending r235 replacement package only, bind the observed Phase-1 Kaggle submission envelope (11.8 GiB HDD, 12.2 GiB RAM, 2 vCPUs, 197.7 MiB archive limit) and replace the old eight-lane viability topology with exactly two isolated simulator/search lanes on one child-owned shared tree. | Staged for fresh exact-package, saved-replay, full-game, resource, and immutable-binding receipts. Retain r234 bounded broker/direct fallback, complete 65,536 enumeration, r236 libcg, exact R235 label, and the single-upload limit. Historical r228 evidence and r229's r233 outer watchdog/lifecycle separation remain unchanged. No upload, restart, training, or BO1000 action. Typed source: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 239-BO1000 | 2026-08-10 | For the next r229 BO1000 package only, replace the former eight-lane shared-tree topology with exactly two isolated simulator/search lanes; the sealed eight-lane r236 package is preflight-only/ineligible and started no game. | Preserve r230 65,536 enumeration, r233 outer watchdog/requeue/quarantine, r236 official libcg, and the pre-r234 r229 lifecycle baseline. No r234 Kaggle broker/direct fallback/queue cleanup, managed-service restart, training, Kaggle action, or fleet execution is authorized until new two-lane package/preflight receipts pass. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 240-KAGGLE | 2026-08-10 | For the pending r235 Kaggle replacement only, return the exact precomputed legal frozen-r195 direct action immediately when every selected factorized-stage probability is finite and ≥0.90; otherwise use exactly two-lane MCTS with 2.0s child / 4.0s parent limits, qualified early stop (≥8 backups, same leader ×3, both lanes progressed), and 32-backup hard stop. An ambiguous-MCTS receipt may carry a deterministic continuation plan of ≤8 actions, consumed only after exact fingerprint/actor/legal-order/two-lane-backed-leader/no-transition validation. | High-confidence direct decisions are journaled mode `high_confidence_frozen_direct`, never start/call a child, and are not degraded. A valid deterministic plan has proven-step precedence, journals exactly once with planned-vs-direct, and rewrites history to its actual action; any mismatch clears it and returns to normal high-confidence/adaptive MCTS. Zero backups retains the existing precomputed-direct fallback/containment rule. Preserve r234, r236, 65,536, r238 resources, exact R235 single upload/label, and historical old 8s evidence; no r229/BO changes, jobs, upload, restart, training, or selector authority. Typed source: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 241-TRAINING | 2026-08-10 | Train a separately versioned direct-policy Alakazam successor from the immutable r195 checkpoint on the owner's exact new 60-card list and supplied guide: exactly ten updates, 1,024 self-play plus ≥1,024 pinned direct H10 Marnie games per update, and five-epoch expert soft refreshes after updates five and ten using the exact rolling 2026-07-22..08-10 corpus. | Staged and fail-closed until the unpublished August 10 expert day and every exact-deck, official-r236-libcg, direct-Marnie, fixed-cycle, terminal-refresh, and policy-only-package receipt pass. Submit exactly once, first-preferring, only after `iter_00009` plus `expert_before_iter_00010.pt`; no MCTS/RTP/Guide2Vec/search targets, eleventh collection, intermediate submit, retry, or copy. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 242-KAGGLE | 2026-08-10 | Supersede only the r240 high-confidence frozen-direct threshold for the pending r235 replacement: every selected factorized-stage probability must be finite and ≥0.80 inclusive, otherwise route to unchanged ambiguous two-lane MCTS. A qualified direct starts no child or MCTS/select/search/model/simulator call; an existing child receives only one history-only `note_direct_action` IPC. MCTS rollout expansion stops at terminal/chance/away-from-root-actor boundaries, where the leaf is value-evaluated without opponent action selection/planning. | The prior r240 ≥0.90 threshold draft and any preflight bound to it are historical/ineligible; reissue the existing high-confidence/adaptive-MCTS regression at ≥0.80. Preserve r234/r236/65,536/r238/R235 label-and-single-upload/continuation/adaptive-stop constants and no r229/BO change, job, upload, restart, training, or selector authority. Typed source: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 237-INSPECTOR | 2026-08-10 | Remove the Replay Model Inspector's 20-second selected-trace browser cutoff and let Elmo finish the exact checksum-bound archived-model forward reconstruction. Keep Bert transport-only, preserve selected-trace-only loading/no whole-game prefetch, and abort only requests made stale by a newer browser selection. | Activated on Elmo at `2026-08-10T23:19:01Z` through an inspector-only managed restart. Focused static/inference/server tests passed 23/23; exact live cold and resident-model reconstructions exceeded the removed cutoff and returned HTTP 200, while a different runtime package also completed its checksum-bound isolated forward with verified provenance. Retain one serialized resident GPU model, isolated-worker fault containment, read-only replay/checkpoint bytes, `recomputed_not_historical`, and zero training/selector/submission authority. Typed source: `state/replay-model-inspector-forward-pass-reconstruction-r237.json`; receipt: `state/replay-model-inspector-forward-pass-reconstruction-activation-r237.json`. |
| 243-INSPECTOR | 2026-08-10 | Materialize every baseline trace for the explicitly selected physical replay game once on Elmo, reuse one causal model state across its factorized stages, and make step/stage navigation read or join that game-scoped cache instead of starting another reconstruction. | Implemented and locally validated; live inspector/gateway activation and capacity receipt are pending. The disposable cache is private, whole-game-LRU bounded, checksum-bound, and atomic under Elmo `/tmp`; cache identity binds exact replay/model/runtime/parity inputs, while Playground scales remain separate. One nonresident worker streams cache-independent rows with heartbeat idle containment and no total deadline. Bert remains transport-only and gives exact trace headers no wall-clock deadline while retaining bounded connect/body/concurrency/size controls. Focused r243 suite passed 61/61 and the final gateway/Caddy focus passed 13/13; full Inspector/dashboard suite passed 156 with only the unchanged unrelated r187 recipe-registry test failing. Preserve read-only artifacts and no training/selector/submission authority. Typed source: `state/replay-model-inspector-physical-game-materialization-r243.json`. |
| 244-LIBCG | 2026-08-11 | Correct official-libcg SearchId identity only for the pending r235 Kaggle replacement and next r239 BO1000 package: numeric SearchId is scoped per distinct `AgentStart` handle, so both first raw IDs may be `0`. Require exactly two arenas/SearchBegin calls, two per-lane handles and chains, and two distinct `(handle_identity, first_search_id)` states; global raw-ID uniqueness is not required. | Staged for fresh handle-scoped identity receipts and immutable bindings. Preserve r242 Kaggle scheduling/containment, r239 two-lane topology, r233 watchdog/pre-r234 BO lifecycle, r236 library identity, all action/resource/upload boundaries, and historical evidence. No code/job/upload/commit, service, training, selector, or game authority is added. Typed sources: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`, `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 245-TRAINING | 2026-08-10 | Clarify r241: preserve peak-r195 behavior while changing only the exact deck, supplied guide, r236 simulator binding, ten-update horizon, and requested expert refreshes. Keep every architecture-present non-combo head/route live and trainable, combo loss/route off, Matchup Adapters on, and the established diverse 7,172-game public mix with at least 1,024 direct H10 Marnie games. | Canonical immediately for the staged r241 lineage. H10 Marnie is a minimum row, not the sole public opponent; guide curriculum remains training-only and cannot suppress heads, adapters, public opponents, or research controls. Existing source blocker and one-terminal-submit boundary remain unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 246-KAGGLE | 2026-08-11 | For an ambiguous r235 two-lane MCTS decision only, give a single exact stock-simulator proof of a deterministic terminal win reachable this turn absolute root-action and early-stop authority over priors, visits, and nonterminal alternatives. Keep r242's `>=0.80` direct-before-child route unchanged. The proof is valid only when it binds the current root observation/legal fingerprints, actor, legal root action, selected action, and terminal `win` for that actor; every simulated action remains that actor's and crosses no chance, unresolved randomness, actor/opponent boundary, draw, or loss. | Staged for focused terminal-win proof regression and immutable binding. Two lanes still initialize/clean, but dual proof, both-lane progress, normal `>=8`/leader×3 thresholds, and exhaustive scan are not required after one valid deterministic proof. Stale/malformed/loss/draw/nonterminal/chance/opponent claims have zero override authority and follow existing fail-closed classification. Preserve r234/r236/r238/r242/r244/65,536/R235 one-upload rules and all r229/BO boundaries. Typed source: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`. |
| 247-TRAINING | 2026-08-11 | Refresh r241's Matchup Adapter archetype identities from the newest completed authenticated PTCGReplay ingest. Append exact numeric-ID/name identities only to never-used Router Format 6 slots; preserve every existing slot and peak-r195 tensor byte-for-byte, and never treat site meta as action, gradient, outcome, or gate evidence. | Staged pending browser-authenticated snapshot and immutable receipt. New slots start exact-zero/dormant and require checksum-backed replay support plus causal-router fit/precision/support/bypass/runtime receipts before training or activation. Preserve the already-running exact-20 roster-18 corpus jobs and every r241 schedule/deck/guide/direct-policy/one-submit boundary. The credential is runtime-only and never committed or receipted. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 248-TRAINING | 2026-08-11 | Defer the r247 PTCGReplay archetype refresh for now. Run r241 with the existing peak-r195 Matchup Adapter identities/tree unchanged and require checkpoint-derived `no_slot_change`; no new site snapshot, slot allocation, fit, or activation is part of this cycle. | Canonical immediately for staged r241. PTCGReplay and append-only migration receipts are not launch gates; the global roster remains byte-unchanged. Preserve Matchup Adapter runtime ON, the exact deck/guide, direct-only r236 simulator, 1,024 + 7,172 loop, ten updates, two five-epoch refreshes, all non-combo heads/routes, and exactly one terminal submit. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 249-BO1000 | 2026-08-11 | Keep one frozen model and one logical shared MCTS tree in the r229 game process, but move exactly two official-r236 native simulator lanes into two persistent owned child processes. Bound every native request and cleanup; after a lane hang/crash, reap the exact child, discard the failed partial tree, reopen fresh two-lane workers, and retry the same complete-root eight-second search once. | A successful retry remains fully backed two-lane MCTS. Only retry exhaustion may use the precomputed legal same-state r195 direct action, explicitly degraded with zero change credit and complete fault/reap telemetry. Clean full-game preflight requires zero exhausted fallbacks. Preserve r230/r233/r236/r239/r244 and exclude all Kaggle broker, adaptive/direct-bypass/continuation/terminal-win policy changes. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 250-BO1000 | 2026-08-11 | Supersede r239/r249's active two-lane topology with serial MCTS for now: one parent-owned frozen model/tree and exactly one process-owned official-r236 native simulator handle, with no concurrent libcg calls. Keep bounded reap, discard the failed partial tree, reopen one fresh serial child, and retry the complete root once. | Authorized for serial implementation, tests, sealing, and a clean full-game preflight requiring zero exhausted fallbacks. The two-lane packages/results are historical and ineligible. Only after serial passes may a separately versioned process-parallel node-evaluation/tree-building design be investigated; it has no current execution authority and cannot alter Kaggle lifecycle. Preserve r230/r233/r236/r244, the eight-second budget, and all BO/Kaggle separation boundaries. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 251-TRAINING | 2026-08-11 | Correct r241's activation topology: immutable owner intent → checksum-bound source and baseline payload snapshots → offline host receipts → one logical create-only activation overlay → managed services. Owner intent contains no derived readiness/operation authorization. The one overlay binds both hosts and is byte-identically mirrored with one shared SHA. Clarify direct/no-MCTS scope as learner, pinned H10 Marnie, target generation, terminal package, and submission only; preserve frozen non-H10 diverse public packages/selectors unchanged. | Staged. The pending source-snapshot registry cannot authorize execution; only the external overlay's checksum-bound owner-start authorization can. No service/training/submission activation occurred. Preserve exact deck/guide/r236/10-update/1024+7172/refresh/full-head/adapter/PTCG-defer/one-submit requirements, do not add a public-search firewall, and do not alter concurrent MCTS work. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 252-MCTS | 2026-08-11 | For both the pending r235 Kaggle replacement and the separately owned r229 BO1000 experiment, stop private simulation at every recognized chance context, every stochastic resolution with more than ten outcomes or equiprobable outcome probability at most 0.10, and every deterministic internal choice with more than 64 complete ordered outcomes. Each is value-only with zero sampling, action, child, planning, or continuation authority. | Staged for typed-source, runtime, telemetry, regression, and fresh-package validation. Exact real roots through 65,536 remain completely enumerated. Kaggle hard-fails above the cap; r229 BO records `oversized_direct_fallback`, uses only its legal factorized direct action, and assigns zero search/change credit without counting the decision as MCTS. Preserve Kaggle r242/r246/two-lane/containment/R235 single-upload rules and BO r250 serial-lane/r233/r236 lifecycle rules. The sealed r246 Kaggle archive is historical and upload-ineligible; no Kaggle upload is authorized by this record. Typed sources: `state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json`, `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 253-BO1000 | 2026-08-11 | Supersede r250's one-continuous-trajectory serial mechanics with repeated independent root rollouts over one parent-owned logical tree and one process-owned official-r236 handle. Each rollout starts with `SearchBegin` at the exact physical root, follows parent-tree-selected actions to one evaluated/r252-boundary leaf, backs up, boundedly releases/ends, and reopens the root until the decision deadline or rollout cap. | Staged for BO-only typed-source/code/tests/package/full-game replacement. Any rollout fault discards the entire attempt/tree and permits exactly one fresh full-root retry; no successful prefix survives. The stopped 13-game/741-search launch is immutable invalid-diagnostic evidence with zero BO/MCTS-effect authority because only one root edge was structurally visitable. Require constructed multi-root-edge and changed-selection regressions before relaunch. Preserve r233/r236/r252, host-local data plane, and all Kaggle boundaries; Kaggle remains paused and unchanged. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 254-BO1000 | 2026-08-11 | Fully abandon the r253 full-MCTS BO1000 at its clean 45-game stop. Preserve the incomplete prefix as diagnostic-only evidence, never a 1,000-game efficacy result. Replace the next research direction with conflict-triggered conservative tactical proof search: direct r195 is default, and initially only an exact deterministic terminal win this turn may override it. | r253 service is stopped and must not resume; the 955 unplayed games and process-parallel full-MCTS follow-on are abandoned. Selective-search implementation/tests/package preflight and a small training-ineligible shadow pilot are authorized only after a fresh typed trigger/budget/pilot configuration. Preserve chance/opponent/>64 boundaries, correct simulated previous-action history, and zero Kaggle/training/serving/selector/model authority. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 257-BO | 2026-08-11 | Reopen only sidecar implementation and local testing of a shadow-only, goal-directed tactical sequence planner for exact same-turn terminal-win discovery and SME tutor/resource planning. It is policy-ordered limited-discrepancy search, not MCTS; r195 remains the only dispatched action authority. | New sidecar and focused deterministic/offline tests only. Stop at chance, hidden-information/tutor re-observation, actor/turn change, or >64 complete actions; preserve simulated action history. Native adapters require an owned bounded child. Tactical-outcome/SME/model signals are hints, never proof. No package, pilot, fleet game, Kaggle, training, serving, selector, promotion, model, service, or active-refresh change is authorized. Typed source: `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 258-TRAINING | 2026-08-11 | Build a separately versioned direct-policy Alakazam successor with a match-updated causal `OwnDeckLedger` shared by policy, value, every learned head, option decoding, and Fusion; learn real visible deck-search/tutor decisions and add narrowly masked immediate terminal-conversion supervision. | Build and offline-test the isolated dormant successor now. Do not mutate or interrupt r241. Staged migration/training begins only after receipt-proven terminal completion of the Alakazam refresh; production activation then requires causal coverage, zero-safe migration, all-head gradient reachability, local/remote/replay parity, calibration, bounded influence, source-disjoint evaluation, and promotion receipts. Typed source: `state/alakazam-own-deck-ledger-successor-r258.json`. |
| 259-TRAINING | 2026-08-11 | Start Elmo materialization of the r258 causal ledger/tutor/terminal replay inputs now as a separate checksum-bound side store over the exact r241 2026-07-22 through 2026-08-10 expert manifest. Keep source archives read-only and key rows by episode, acting seat, environment step, and observation fingerprint. | Authorized now through the new bounded `pokebot-own-deck-rollout-store-r259.service` only. It may not stop or alter any existing service, may not feed active r241, and becomes next-train eligible only after refresh completion plus join, parity, schema, count, and digest receipts. Typed source: `state/alakazam-own-deck-ledger-successor-r258.json`. |
| 260-TRAINING | 2026-08-11 | Highest priority: because r241 never armed or started and its final launcher check failed closed, fold last night's r258 `OwnDeckLedger` architecture into the next r241 receipt line before update 0. Preserve the 19 inherited heads/18 active Fusion-v3 routes and combo-off semantics; add the width-128 shared ledger, eight-feature option adapter, and typed 7-output tutor-completion plus 6-output terminal-conversion heads/routes with zero-safe migration and a total masked auxiliary budget of 0.05. | Explicitly supersedes r258's post-r241-completion/no-r241-mutation delay only at this safe pre-start boundary. Bind the completed Elmo exact-20-day side store only after all 20 shards and join/parity/schema/count/digest receipts pass; then require the corrected typed parent `FileIdentity`, zero-safe migration, training canary, bounded influence/evaluation, new immutable source/H10/peak/image/quartet/overlay, and managed-service preflight. Activate immediately after those receipts without another owner decision. Old 1c34/v8/v6/r13 artifacts remain immutable inactive history. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 261-TRAINING | 2026-08-11 | Make Inzi the sole managed training host for the r260 combined successor. Elmo may preprocess and receipt the protected expert corpus and run bounded disposable parity checks, but it may not train the learner. | After 20/20 Elmo sidecar completion, create-only copy the exact daily layout and deterministic joined dataset to the canonical Inzi r260 training root, rehash typed FileIdentities there, and require the Inzi trainer to use only its local disk-backed streaming index. Any Elmo `/mnt/Main/` training path fails closed. All r260 architecture, evidence, schedule, direct-policy, and activation gates remain unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 262-TRAINING | 2026-08-11 | Begin transferring already committed immutable r259 daily sidecars to a non-eligible Inzi staging root while Elmo finishes the remaining days. | Transfer only committed non-dot daily directories, create-only and byte-identical; seal and rehash each day on Inzi, append later commits, and retain zero training authority until 20/20 plus join/parity/transport receipts pass and the tree is atomically promoted. Do not restart or reconfigure the healthy r259 producer. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 263-TRAINING | 2026-08-11 | Train the revision-257 public-state tactical sequence planner's outcome hint live as a shadow cotrain objective in the same ten-update r241 successor, and explicitly train the full causal OwnDeckLedger/tutor/terminal sidecar surface in every update and applicable expert refresh. | Add one zero-safe 4-output tactical outcome hint head at masked loss 0.025 while preserving the deck auxiliaries' 0.05 budget. Require at least 1,024 bounded tactical shadow roots per update and exact trace/accounting/parity/influence/latency receipts. The ordinary direct policy remains the sole dispatched and submitted action; no MCTS, RTP, tactical override, hidden-state target, evaluation/Kaggle training, selector, or serving authority is granted. Typed source: `state/alakazam-new-list-direct-policy-r241.json`; revision-257 planner provenance remains `state/alakazam-r228-vs-r195-no-mcts-fleet-bo1000-r229.json`. |
| 264-TRAINING | 2026-08-11 | Use the immutable ladder-proven r195 model checkpoint as the zero-safe weight parent, then run the established full Alakazam cycle: exact 25-epoch expert bootstrap and submit, followed by exactly 25 RL updates in five-update blocks, each block ending in an exact five-epoch soft refresh and submit. | Exactly six first-if-allowed receipt-backed direct-policy submissions are authorized: bootstrap plus boundaries 5/10/15/20/25. Preserve 1,024 self-play + 7,172 public mix, ≥1,024 H10 Marnie, and 4,098/4,098 seats per update. Cotrain r263 deck/tactical objectives through all 25 updates. The r195 RTP sidecar remains excluded from action/target/package/serving authority even though its 600.0 submission proves the shared parent model lineage. No iter_00025 collection, skip, duplicate, or unreceipted retry is allowed. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 265-TRAINING | 2026-08-11 | Make every ordinary architecture-present head live and trainable in the r264 successor, including activating Alakazam `combo_state` with masked loss 0.025 and its inherited Fusion route. | Require all 19 inherited routes plus the two r260 option routes active, with per-head support/gradient/optimizer/checkpoint/influence receipts. The r263 tactical outcome head is also actively trained and used as a shadow ordering hint, but remains outside action Fusion so it cannot dispatch or override direct-policy actions. No hidden/fabricated label or dead-head placeholder is eligible. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 266-TRAINING | 2026-08-11 | Activate the learned tactical-sequence outcome head's direct-policy Fusion route so its learned public-state representation affects the learner, while keeping the bounded sequence search planner itself shadow-only. | All 22 heads and 22 routes are trainable/active. Require route-on/off paired ablation at bootstrap and every five-update boundary with action-change rate, margin/KL/value deltas, calibration, support, magnitude, and latency, plus bounded nonzero influence and source-disjoint outcome receipts. Show the isolated tactical impact on the dashboard. The planner still cannot dispatch/override actions or treat hints as proof. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 267-TRAINING | 2026-08-11 | Add the exact frozen r195 600.0 ladder submission package to the public specialist research roster for at least 128 games in every update. | Count the cell inside the fixed 7,172 public games, require at least 64/64 learner seats, and preserve ≥1,024 direct H10 Marnie games. The r195 RTP sidecar may act only for that frozen opponent; no r195 weight/RTP/action/trace/hidden state enters learner targets, runtime, or submissions. Receipt and dashboard-report package identity, counts, seats, and outcomes separately each update. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 268-TRAINING | 2026-08-11 | Correct the r267 research opponent to the owner's exact linked Kaggle submission `55378392`, the frozen r195 NO-RTP direct-policy bundle `sha256:dfa8bfcc…b7145` (500.4). | Submission 55378477, its `2f982f25…` bundle, and RTP sidecar are excluded. Preserve ≥128 games/update, ≥64/64 learner seats, fixed 7,172 public total, ≥1,024 H10 Marnie, per-update receipts, and dashboard reporting. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 269-TRAINING | 2026-08-11 | Before training, use Elmo and the newest completed authenticated PTCGReplay ingest to prepare new Matchup Adapter identities/routes, including a distinct frozen 55378392 NO-RTP research-opponent route. | Supersedes r248's deferral. Seal the source snapshot and append only to never-used Format-6 slots starting at 20; preserve slots 0--19 and r195 tensors byte-identically. New slots begin exact-zero/dormant and require data-support, causal-fit, precision/support, bypass, disjointness, and readiness receipts before training, with runtime activation separately gated. Use separate managed Elmo workloads and do not alter healthy r259. PTCGReplay meta has zero action/target/outcome/gate authority and credentials stay runtime-only. Bind all readiness receipts into the one pre-start overlay; preserve r263--r268 deck/guide/all-head/tactical/25-update/six-submit requirements. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 270-TRAINING | 2026-08-11 | Also collect the decklists exposed by the completed PTCGReplay Meta snapshot to help prepare the new matchup adapters. | Decklists are guide/reference evidence only: soft card-signature features, replay discovery, confidence diagnostics, and review. They are not exact deck requirements, hard rules, labels, eligibility/gate evidence, or routing proof. Independent checksum-backed replay support and all r269 receipts remain mandatory. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 271-TRAINING | 2026-08-11 | Retrain a separately versioned causal Matchup Adapter router candidate on Elmo for the expanded r269/r270 Meta snapshot roster before r241 training. | Preserve the existing router/crosswalk, slots 0--19, and r195 tensors immutably. Fit only on checksum-backed public-state replay observations with source-disjoint splits; snapshot/decklist metadata is guide/identity/stratification evidence only. Require per-ID and aggregate precision/support, ambiguity/unknown bypass, balance, calibration, collision, latency, parity, crosswalk, and distinct-55378392 receipts. Activation is overlay-only after all gates pass; failure blocks training without threshold weakening. Do not touch r259 or interactive sessions. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 272-TRAINING | 2026-08-11 | Add every real archetype from the completed PTCGReplay Meta snapshot window, not only the top-20 rows. | Exclude `Rogue / Other`; preserve all existing slot identities; append each unseen numeric ID once in stable snapshot order. Unsupported additions remain exact-zero/dormant until r269/r271 fit and activation gates pass. Keep frozen submission 55378392 a distinct non-aliased opponent identity. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 273-TRAINING | 2026-08-11 | Send every snapshot row, guide decklist, replay-support index, router target, and adapter allocation to its exact PTCGReplay numeric ID. | Names are display-only and cannot merge or alias IDs. Current Teal ID 167 stays distinct from preserved historical ID 151. Kaggle submission 55378392 remains a separate package identity, not a PTCGReplay ID. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 274-TRAINING | 2026-08-11 | Complete the already transferred twenty-day sidecar dataset, all pre-start gates, overlay, and managed trainer setup locally on Inzi without retransmitting the large payload. | Rehash all Inzi daily files against preserved immutable per-day receipts, ignore dot-prefixed failed-transfer remnants, and issue a new local-post-transfer attestation before deterministic join/promotion. Elmo stays read-only source/receipt truth and may send only compact evidence and adapter artifacts. A later low-priority create-only return copy is archival-only and cannot gate or interrupt training. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 275-TRAINING | 2026-08-11 | Convert the never-started new training runtime from candidate r241 to candidate r274. | All active run, unit, registry, overlay, receipt, package, and dashboard identities use r274. r241 remains design/transfer provenance only and has no runtime authority. Keep the large transferred r260 payload at its existing checksum-bound physical path and bind that imported provenance into r274 rather than copying it for naming alone. Preserve the full r263--r273 feature and schedule stack. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 276-TRAINING | 2026-08-11 | Use checksum-gated four-way LibcgMultiEnv packing to accelerate the r274 simulator loop and keep Inzi's available simulator cores engaged. | Pack self-play and only exact r182-retention-compatible public packages; every unknown, changed, legacy, malformed, research, or unattested package remains singleton. Preserve independent child accounting, remote participation, exact 8,196 games, 4,098/4,098 seats, ≥1,024 H10 Marnie, ≥128 exact submission 55378392, direct-policy-only semantics, and all existing gates. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 277-TRAINING | 2026-08-11 | Shadow-train the new tactical-sequence outcome head during the 25-epoch bootstrap, then use its learned route only after bootstrap. | Bootstrap route influence is exact zero and the bootstrap submission excludes it. Activate the zero-safe Fusion route only after checksum-backed bootstrap/submission plus gradient, coverage, calibration, bounded-influence, parity, and impact receipts; use it for RL updates 0–24 and later refreshes. The bounded search planner remains shadow-only throughout. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 278-KAGGLE | 2026-08-11 | Submit one raw r195 NO-RTP test with only `deck.csv` replaced by the exact new r241 Alakazam list, visibly labeled `r195 NO RTP raw resubmit, new Alakazam list, not retrained otherwise`. | No retraining or model/runtime/adapter/search/selector/registration/training-service change. Preserve every non-deck archive member byte-for-byte, use one-shot `first_if_allowed`, and keep all r274 training plus six-submission obligations unchanged. Typed source: `state/alakazam-r195-new-list-no-retrain-r278.json`. |
| 279-TRAINING | 2026-08-11 | Permanently supersede the stopped resident-Python-object r274 bootstrap loader and materialize one immutable joined contiguous expert pack locally on Inzi. | The sealed pack must bulk-join OwnDeckLedger once, attach tactical labels once, encode flat numeric arrays plus offsets, validate exactly 26,704 Alakazam games and 2,040,911 decisions, and be reused for the 25-epoch bootstrap and later five-epoch refreshes. The old object loader remains stopped and ineligible. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 280-TRAINING | 2026-08-11 | Keep the revision-279 CPU pack as restart truth but make full numeric residency on Inzi `cuda:1` the primary epoch path. | Copy the validated pack once, gather game batches on-device, and avoid ordinary host decoding or repeated CPU-to-GPU batch streaming. Pinned CPU streaming is fallback-only after measured allocation/headroom failure; the Python-object loader is never a fallback. This changes no model, target, route, adapter, schedule, deck, RTP, or submission contract. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 281-TRAINING | 2026-08-11 | A bootstrap with zero Matchup Adapter rows is incomplete: train the eligible adapter bank before the bootstrap submission and RL update 0. | Run a checksum-bound adapter-only optimizer phase from training-eligible, exact-identity, source-disjoint replay support. Require nonzero rows/steps, finite changed eligible adapter tensors, per-route validation, non-adapter bit identity, and exact-zero dormant unsupported slots. Preserve the completed ordinary/tactical parents; the submission must descend from the adapter-trained child. The one-epoch adapter continuation after each later RL update remains required. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 282-TRAINING | 2026-08-12 | Remove per-update tactical shadow-search materialization from r274 and resume the sealed update-0 collection directly into full-model backprop. | Preserve bootstrap tactical evidence, but use zero tactical-sequence-specific RL loss and do not recollect update 0. All other retained heads, adapters, counts, and schedule remain unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 283-TRAINING | 2026-08-12 | Physically remove only the new tactical-sequence head and route before update-0 backprop. | Preserve the old strategic tactical head and every other retained head/route. The active learner has 21 heads/routes and resumes the sealed collection without recollection. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 284-TRAINING | 2026-08-12 | Move the replacement formal holdout to after update 1, restrict every future formal holdout to exact frozen r195 submission 55378392 plus exact H10 Marnie, and deactivate combo again. | Preserve the stopped 2,142/4,500 broad-holdout prefix as diagnostic-only and never call it a pass. Advance the trained update-0 candidate through one immutable deferral receipt; formal holdout becomes exactly 500 games (250/opponent, 125/125 seats) after update 1 and later. Keep the diverse public training mix separate. Retain combo tensors but set combo loss zero and route off from iteration 1 onward. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 285-TRAINING | 2026-08-12 | Run exactly 20 r274 RL updates, not 25. | Commit iterations 0–19, refresh and submit at 5/10/15/20, and forbid iteration-20 collection. The bootstrap plus four RL-boundary submissions make exactly five; the former boundary-25 obligation is removed. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 286-TRAINING | 2026-08-12 | Quarantine Elmo from active r274 production until it is independently fixed and canary-proven. | Cleanly stop only the managed trainer, seal iteration 1 for append-only partial resume, and resume with Bert as the sole remote while preserving completed games and every training invariant. Elmo requires exact singleton/four-game import, child-wide recurrence, checksum, and useful-throughput evidence plus explicit owner readmission before rejoining. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 287-TRAINING | 2026-08-12 | Resume the r274 production loop on Inzi only for now. | Preserve the exact sealed iteration-1 prefix and checkpoint, disable every remote endpoint, and collect only missing jobs with 32 local simulator workers. Elmo repair remains isolated; Bert is healthy but intentionally excluded. RTP/search remain off and all counts, holdouts, adapters, refreshes, and submission boundaries remain unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 288-TRAINING | 2026-08-12 | Stage a default-off bounded Alakazam turn-checklist logit residual covering the eight named causal questions, bound to the supplied checklist attachment and exact new-list deck rather than treating r195 as the identical list. | Apply only after current neural/Fusion/OwnDeck/Matchup Adapter effects and before legal selection; clip total residual to ±0.10, neutralize unknown/malformed/noncausal evidence, preserve guide runtime authority false, and forbid hidden inference/search/RTP. Elmo may calibrate only nine scalar gates against source-disjoint exact-new-list data with all neural tensors frozen; this is isolated validation and cannot rejoin production. Do not touch the active iteration; activate only at the next safe receipt-backed boundary. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 289-TRAINING | 2026-08-12 | Permit one isolated Elmo diagnostic BO250: 125 seed-matched seat-swapped pairs using the exact new list on both arms, comparing the receipt-bound r274 checklist candidate with immutable r195 NO-RTP direct control. | Require exact 125/125 candidate seats and 125/125 actual-first/second, source-disjoint calibration seeds/rows, direct-policy-only execution, and a fail-closed exact r274 artifact/parity check. It is training-ineligible, non-promotion, has no production/selector/Kaggle/readmission authority, and cannot alter the active iteration. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 290-TRAINING | 2026-08-12 | Make the r288 checklist layer a staged component of the current Inzi r274 learner lineage, not an r195-only side model. | Activate only at the first immutable iteration boundary after the in-flight collection seals and before next dispatch—never mid-shard/current iteration/recollection/restart. Thereafter use it consistently for r274 collection, direct/refresh evaluation, and later receipt-built submission runtime. Neural/head/OwnDeck/Adapter training stays unchanged; Elmo calibration/BO250 stays isolated and reaches Inzi only through a checksum-bound parity receipt. RTP/search/MCTS and production Elmo remain off. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 291-TRAINING | 2026-08-12 | Supersede only r290’s activation timing: retain the checklist as an additive r274 candidate, but do not deploy or activate it on Inzi at any current boundary. | Test only in an isolated Elmo non-production directory/process using create-only/read-only exact receipt-bound artifacts. Do not alter Inzi files/services/runtime or active trainer, including boundary config/drop-ins; no production-worker change. Complete local+Elmo unit/parity/calibration and BO250 first. A later Inzi activation requires a new explicit owner decision; neither results nor a boundary can auto-activate it. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 292-TRAINING | 2026-08-12 | Correct `replacement_alakazam_line` to mean only the bench backup attacker; exclude the current Active Alakazam line. | Trace `ready` only for Benched Alakazam plus Psychic-providing Energy; trace `completable` only for Benched Kadabra/Abra with all exact visible evolution, Energy, and timing resources; mark unknown draw/possibly prized dependency `not_live`, and unknown eligibility/timing neutral/unavailable. Require a regression that Active Alakazam alone does not count. Preserve r291 Elmo-only/no-Inzi isolation and delay BO250 until corrected local/Elmo regression and parity checks pass. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 293-TRAINING | 2026-08-12 | Audit checklist overlap against active r274 heads/Fusion/OwnDeck/Matchup Adapter, leaving existing logic unchanged by default. | Resolve duplicate influence only in new-layer gates/traces; changes to existing learned logic require separately evidenced/reported drastic correctness issue. Legacy broad guide scoring has no runtime residual authority while semantic contradictions remain; attachment rules may inform the eight channels, while the separate guide gate is exact-zero or trace-only until source-disjoint corrected validation. Preserve r291 Elmo-only/no-Inzi and delay BO250 until the r293 audit/guide-gate receipt passes. No code/service change is authorized. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 294-TRAINING | 2026-08-12 | Bind attachment `fdee1eae…` / `458019…278a3` as the superseding guide-scorer semantics source. | Update only the training-only `alakazam_new_list_heuristics` scorer later; retain neural/Fusion/OwnDeck/Adapter logic. The attachment's then-incomplete four-Alakazam sentence had no deck authority, so the checksum-bound canonical list stayed unchanged. Broad guide support remains zero/trace-only; preserve Elmo-only/no-Inzi. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 295-TRAINING | 2026-08-12 | Supersede r294’s attachment byte identity after the same attachment was corrected in place to `5cc092…dff8b`. | The corrected owner-supplied exact 17/36/7 inventory, including three Alakazam, matches the canonical checksum-bound list; leave its bytes/digest unchanged. Preserve the scorer-update/training-only/no-direct-runtime-action boundary, learned-head immutability, broad-guide zero/trace-only gate, and Elmo-only/no-Inzi isolation. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 298-TRAINING | 2026-08-12 | Stage an isolated Elmo-only simulator-rules representation and auxiliary-head overhaul from attachment `7b1d4464…` / `d3f060…d2cb`. | Preserve immutable r195 and exact r241/r274 baseline. Build only a versioned zero-gated derivative with exact layer-off logits; do not change/restart/activate/package/propagate Inzi or production. The simulator/legal options and public information set are authoritative; Phase 1 collision census, Phase 2 public-rule representation, Phase 3 simulator targets/head repairs, Phase 4 eight-channel trace contract, and Phase 5 engine/metamorphic tests are mandatory. Q3 remains Bench-only and Q5/Q6 remain trace-only with exact-zero gates pending separate calibration. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 299-TRAINING | 2026-08-12 | Rescind r297's hard 32-process Inzi floor; retain four environments per selected process and bias inference leaves toward Blackwell. | Leave the current 32×4 collection untouched until its next clean receipt-backed boundary. At that boundary select process count from sustained completed-game/resource evidence, make Blackwell the primary leaf target, and retain the 3080 Ti as bounded spillover. Inzi-only and RTP/search/MCTS-off remain unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 300-TRAINING | 2026-08-12 | Select exactly 16 simulator processes × 4 environments for the next Inzi production topology. | Activate 16×4 only after the current self-play phase seals; make Blackwell primary for policy-leaf inference with the 3080 Ti as bounded spillover. Preserve the shard, Inzi-only operation, and RTP/search/MCTS-off. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 301-TRAINING | 2026-08-12 | Restore 96 concurrent local games using the packed equivalent of the historical 96×1 topology: exactly 24 processes × 4 environments. | Supersede r300's 16×4 selection. Activate 24×4 after the current self-play phase seals, with Blackwell primary and the 3080 Ti as bounded spillover. Preserve the shard, Inzi-only operation, and RTP/search/MCTS-off. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 302-TRAINING | 2026-08-12 | Hand revision 298's isolated Elmo Alakazam rules/head experiment to its own authoritative goal and sole typed source, bound to attachment `8100a736…` / `0f440f…af6f`. | Preserve r298 as historical provenance. This root and the production r241 state retain only a non-authoritative handoff/reference. The dedicated goal permits isolated Elmo implementation/evaluation only and grants no Inzi, runtime, restart, collection, checkpoint, selector, packaging, submission, propagation, or production authority. Revisions 300--301 are unchanged. Dedicated gateway: `goals/alakazam-elmo-rule-derivative/GOAL.md`. |
| 303-TRAINING | 2026-08-12 | After the derivative's exact Elmo schema/corpus/census and verified staging gates pass, supersede the current Inzi r274 training lineage at one clean receipt-backed boundary with frozen-parent bootstrap/training on Inzi Blackwell; then queue one checksum-exact single-use `first_if_allowed` Kaggle package and, once its accepted-or-quota/spacing-pending receipt is durable, start derivative self-play across the receipt-proven full available fleet without waiting for score. | Preserve healthy r274 until all in-flight collection/shard/optimizer/adapter/checkpoint/commit work seals; then pause only its exact trainer and r274 submission-boundary units via user systemd, keep the shared Kaggle queue service unchanged, preserve every r274 byte and rollback receipt, and forbid concurrent old-lineage work. Staged shards become eligible atomically at handoff. Require exact parent/corpus/schema/frozen-tensor/Blackwell/readiness/activation/rollback, package/queue/upload, fleet inventory/parity/routing/activation/shard receipts. No runtime/service action occurs while recording this revision and no serving selector is authorized. Typed production handoff source: `state/alakazam-new-list-direct-policy-r241.json`; dedicated semantics: `goals/alakazam-elmo-rule-derivative/contract.json`. |
| 304-TRAINING | 2026-08-12 | Finish the active r274 iteration 1, upload its exact durable updated learner to Kaggle, then stop this root task's r274 loop. | Preserve and finish the exact collection, optimizer, adapter epoch, r284 500-game holdout, and `iter_00001` commit. Upload exactly one new first-if-allowed direct-policy/RTP-off submission with immutable request/package/authorization/attempt/upload receipts. Fence before any iteration-2 dispatch, stop the managed r274 loop only after accepted upload, and cancel every later r274 update/refresh/submission obligation. The separate derivative remains owned by its own goal. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 305-TRAINING | 2026-08-12 | Repeat deterministic contiguous packing for every future sealed pure-RL replay set before baseline preparation/backprop, including packed Matchup Adapter continuation metadata; always execute this step locally on Inzi. | Source-only implementation now; do not alter/restart the active revision-304 optimizer. Support 1/2/4/8/16/32 deterministic workers, benchmark selection on Inzi, canonical source order, exact tensor/batch/optimizer parity, cached reuse, and serial fail-safe. Elmo/Bert/LAN are excluded; activation waits for a future sealed receipt-backed boundary. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 306-TRAINING | 2026-08-12 | Use exactly 16 local Inzi workers for the future contiguous RL replay packing step. | Activate on subsequent training starts only; no current-run restart or reconfiguration. Other supported counts remain diagnostic-only and production auto-selection is disabled. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 307-TRAINING | 2026-08-12 | For the terminal r274 iteration 1, finish the active ordinary optimizer and one Matchup Adapter continuation epoch, skip formal holdout and research-control evaluation, then commit, submit to Kaggle, and stop before iteration 2. | This exact iteration receives an owner evaluation waiver and may not be called a measured pass. Preserve earlier evidence, do not interrupt or recollect, keep RTP/search/MCTS off, require the accepted submission-ID receipt, and leave future defaults and the separate derivative goal unchanged. Typed source: `state/alakazam-new-list-direct-policy-r241.json`. |
| 308-TRACKING | 2026-08-12 | Route `/goal` tracking for the Alakazam Elmo rule derivative through its dedicated human gateway, sole typed contract, and concise receipt-backed execution-status projection, while preserving the root/r241 ownership of the separate r274 terminal lineage and production handoff. | Active immediately as a metadata-only routing change. It changes no derivative semantics, runtime, service, selector, checkpoint, corpus, training, staging, queue, or submission authority. |
| 309-KAGGLE | 2026-08-12 | Submit one additional checksum-exact copy of immutable r195 NO-RTP submission `55378392` with exact label `r195 iter 21 261d367e131e NO RTP`. | Activated immediately without changing archive bytes, training, or runtime. Fresh one-shot authorization was consumed before upload; Kaggle submission `55468965` was accepted for evaluation at `2026-08-12T23:38:18Z`. Typed source: `state/alakazam-r195-no-rtp-additional-copy-r309.json`. |
| 310-TRACKING | 2026-08-13 | Record the derivative owner's one-GiB final content-addressed shard and transfer-object limit in the dedicated revision-8 goal/contract. | Pointer-only root reconciliation. The active revision-7 Elmo/Inzi re-featurization runs continue without restart or recomputation and remain predecessor evidence; private partials remain ineligible. This changes no r274, service, selector, training, submission, or activation authority. |

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
- Final-format Alakazam must set `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`. Elmo, and
  Bert after its isolated Apple benchmark, are eligible for self-play, public
  mix, holdout, and evaluation. A compatible idle remote during one of these
  phases is a scheduling defect, not reserved capacity.
- Blackwell is hard-stuck at exactly 96 local simulator workers and 96 local
  games in flight under revision 70. Its worker floor, target, ceiling, and
  live-pool maximum are all 96. No adaptive
  scheduler, memory-pressure rebalance, comparison trial, or automatic
  fallback may lower the worker count. This remains mandatory through entry
  into the first final-submission-format Alakazam model and has no automatic
  expiry there. The existing cgroup, no-swap, and host memory guards remain
  fail-closed infrastructure protections and may stop a failed attempt, but
  they may not silently substitute a smaller profile.
- Every generated or formal-evaluation wave releases its phase-owned local
  simulator pool after all producers have durably queued their results and
  before a remaining result buffer is compacted. A prior wave's workers may
  never remain resident through self-play, public-mix, promotion,
  research-control, or formal-holdout result drain.
- Never terminate or interfere with an interactive SSH, Codex, terminal,
  editor, Cursor, Grok, or Claude session.
- Remote checkpoint publication is idempotent by immutable digest. A fresh,
  complete controller-and-leaf health proof may reuse an already resident,
  pinned exact digest without deserializing it again. Missing, unhealthy, or
  mismatched evidence fails closed to the normal verified reload, and a new
  digest must never be silently treated as resident.
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
- Marnie's retired current-deck guide is optional offline shadow evidence only.
  Its live target generation and loss weight remain off/`0.0`; it has no
  gradient, fusion, action, serving, selection, gate, submission, authorization,
  or blocking authority. Missing or failed shadow evidence never blocks the
  authoritative non-guide path.
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
- Plain `dragapult`, `dragapult-blaziken`, `dragapult-dudunsparce`, and
  `walrein` are not required
  specialist targets and never count toward program completion. Historical
  artifacts and stable matchup-slot evidence remain immutable and non-planning.
- Crustle remains an active causal matchup-router identity while its new H10
  specialist is prepared and trained after Marnie. Its Matchup Router Format 6 slot,
  classifier/crosswalk identity, and materialized adapter route remain
  eligible when observed opponent state resolves to Crustle; the route may
  not be disabled, reindexed, deleted, or reused as a consequence of revision
  61.
- The inference-only public opponent `pilkwang-meta-20260708` is the canonical
  Crustle baseline/practice opponent. Its exact registered package digest is
  reused; it receives no gradients and is not the new specialist checkpoint,
  completion credit, or submission candidate.
- The remaining post-Teal specialist order is exactly Archaludon ex, then
  Slowking. Slowking occupies plain Dragapult's former required-fleet slot, and
  its exact pictured card engine—not a generic Slowking variant—owns its guide,
  representative, corpus filter, and pre-stage identity. After Slowking,
  proceed to the separately gated post-fleet/final-submission preparation
  sequence; there is no third specialist fallback.
- Slowking is a combo/toolbox specialist. Its immutable implementation,
  exact-corpus, resident-pack, and parameter-inventory receipts may authorize
  supervised candidate creation only. It cannot become freeze-, registration-,
  selector-, or RL-launch-ready until its candidate-bound final receipt proves
  causal learned-head and bounded action-route coverage for
  top-deck construction, copied-attack choice, combo pieces, energy movement,
  recovery, bench continuity, disruption response, prizes, and outcome timing.
- Slowking may exceed the ordinary two-million-parameter soft target for
  evidence-backed combo capacity, but its initial specialist-scoped hard
  ceiling remains 3.5 million with exact module accounting and resource,
  causality, gradient-use, ablation, and package receipts.
- The required fleet retains exactly 15 historical slots, but revision 79
  terminally disposes Slowking's slot as `failed_experiment`: 14 specialists
  are frozen and Slowking receives neither passing status nor completion credit.
  This explicit owner exception is sufficient only for releasing the final-format
  Alakazam phase. The post-fleet Alakazam
  and Marnie's Grimmsnarl ex refreshes are separately versioned derivatives,
  not new roster rows and not replacements for their immutable historical
  passing checkpoints.
- Once the revision-79 terminal-fleet receipt proves 14 frozen specialists plus
  the explicit failed Slowking disposition, run the refreshes strictly as
  Alakazam then Marnie's Grimmsnarl ex. The Alakazam refresh is the first
  final-submission-format model and starts immediately after Slowking's
  checksum-bound failed-experiment boundary. Prefer a validated hot start from the immutable
  existing Alakazam checkpoint; if that compatibility migration fails, record
  the failure and use the newest checksum-accepted core through the ordinary
  same-archetype refresh path before expanding to the final format. Resolve all
  current canonical training structures at each refresh start; never pin
  either refresh to today's core version, compatibility derivative, schema
  digest, or runtime package, and never partially overlay tensors from
  incompatible lineages.
- The post-Slowking-disposition final-format Alakazam refresh uses an exact 50/50
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
  and the separately versioned H10 + RTP Slop Box specialist are truthfully
  training-complete under either a measured gate pass or an explicit owner
  ceiling acceptance, then freeze and register. Ceiling acceptance must
  preserve the failed gate evidence and must never be labeled as a measured
  pass. Abandoned Crustle is preserved but is not a current population
  blocker until the owner restores it.   Slop Box H10 + RTP is the active
  specialist path under revisions 170–173; revision 171 adds owner-ceiling
  proceed-to-Kaggle-and-RL when Chao-hard overfits with held still short of 0.90;
  revision 173 adds real Slop Box `combo_state` targets for the retained 32-d head.
- The released own-model population has exactly 15 trainable logical members,
  derived from the checksum-bound frozen registry. Slowking is not a trainable
  population member. Crustle uses its newly completed H10 specialist rather
  than its public baseline. Alakazam and Marnie's Grimmsnarl ex use their
  newly completed H10 final-format bundles as the current versions; their
  original immutable frozen packages remain selected history. The other 12
  members begin from their immutable frozen packages. External agents remain
  research-only. Each member runs the unchanged exact 5-RL/5-rehearsal cycle,
  and learner batches remain capped at the proven-safe 2,048 decisions.
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
