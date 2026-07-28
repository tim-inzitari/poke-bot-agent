# Alakazam matchup adapters: iteration-15 staging contract

Status: staged locally, not deployed, runtime disabled.

The seven `96 → 8 → 96` residual matchup adapters have two deliberately
independent control planes. Completing iteration 15 authorizes an isolated
offline fit from the exact committed learner; it does not authorize runtime
routing or promotion into the live learner.

## Exact boundary

The only exact-boundary receipt is created after `commits/iter_00015.json` and
`loop_state.json` both say `last_completed_iteration=15` and
`next_iteration=16`, but before any iteration-16 collection, shard, metrics,
evaluation, or commit artifact exists. The parent is exactly the commit's
`learner` path and SHA-256—not the champion, held-out champion, candidate, or
lineage base.

Receipt creation is exclusive and immutable. A stale ledger, changed parent,
non-zero parent adapter output, duplicate receipt, or existing iteration-16
artifact fails closed. Receipt creation never mutates production and never
enables the adapter bank.

## Offline training plane

All seven adapters may be fit offline when each sequence has an authoritative
oracle ticket derived from a pinned package or full-deck manifest. The ticket
binds the package/deck digest, canonical matchup route, episode, acting seat,
corpus manifest, and active gate contract. Only Alakazam acting-seat sequences
are eligible.

The corpus is physically and logically route-partitioned. Train/validation
membership is deterministic, route-stratified, and episode-disjoint. Every
route must have non-zero train and validation sequences and decisions.

During fitting:

- the complete parent model is frozen and run in deterministic evaluation mode;
- the optimizer contains only `matchup_adapter_bank.*` parameters;
- each decision uses only its immutable oracle ticket route;
- sparse dispatch gives unrelated experts `grad=None`, so AdamW and weight
  decay cannot modify them;
- policy and value are the only allowed losses;
- base tensors are checked bit-for-bit against the receipt-pinned parent after
  every optimizer step;
- per-route validation metrics are recorded separately;
- source, manifest, gate, implementation, split-membership, and route-order
  digests must match exactly on resume.

The production-scale path is `scripts/stage_matchup_adapter_corpus.py` followed
by `scripts/train_matchup_adapters.py`. The stager performs two verified passes,
keeps episode membership in SQLite, and writes exactly fourteen immutable
shards (seven routes times train/validation). The trainer streams one sequence
at a time and permits only one route in each optimizer batch. Its checkpoint
contract additionally pins AdamW settings, constant-LR/no-scheduler semantics,
batch limits, route order, epoch/early-stop settings, AMP policy, and memory
ceilings. Interrupted mid-epoch runs retain exact aggregate/cursor state and
resume to the same model and optimizer state as an uninterrupted run.

The trainer defaults to an 8 GiB process-RSS ceiling and a 16 GiB available-RAM
floor, checking every batch. A pressure violation writes the last verified
cursor and stops before consuming another batch. A fresh run also refuses to
overwrite any prior latest, best, final, or progress artifact.

An adapter-only checkpoint is never a production learner. Dormant integration
merges only a complete, finite, correctly shaped seven-expert tensor set into a
copy of the exact parent. All seven output projections and all seven validation
partitions must have evidence. The parent optimizer, scheduler, scaler, RNG,
counters, and non-adapter tensors are preserved; the adapter optimizer is
discarded. The merged bank remains disabled.

## Runtime plane

Runtime inference must never receive oracle package, full-deck, corpus, job, or
opponent-archetype identity. It may eventually consume only causal public
prefix evidence. The current evidence artifact supports later gated routing for
three matchups:

- Marnie/Grimmsnarl: public IDs `646`, `647`, `648`
- Garchomp: public IDs `379`, `380`, `381`, `342`, `387`, `1173`
- Rocket's Mewtwo: public ID `431`

Recognition requires two consecutive evidence-bearing states, a unique winner,
full confidence, and a `0.5` margin. Missing current evidence, ambiguity, or an
unsupported-family conflict returns `UNKNOWN_ROUTE` immediately. Search branch
state is copied independently and game boundaries reset the recognizer.

Crustle, Cornerstone Ogerpon, Starmie, and Hammer-Pult remain runtime unknown.
The global iteration boundary cannot override a failed or missing per-route
router gate. Remote leaf/runtime wiring is not implemented, so runtime remains
disabled for every route.

## Current launch blockers

The July 11–20 temporal expert corpus contains approximately 37,390 Alakazam
acting-seat sequences and 3,015,860 decisions, with useful coverage for all
seven matchups. Its compact shards preserve temporal actions but deliberately
discarded per-record package/full-deck provenance. They cannot yet authorize an
oracle route or prove exclusion of active formal-gate packages.

The bounded stager and streaming trainer are implemented and covered by an
interrupted/resumed end-to-end test. Before the real boundary fit can run,
regenerate a checksummed oracle sidecar from the pinned raw archives and join it
by immutable source/archive digest, episode, and acting seat. The sidecar must
also bind every historical opponent identity to an authoritative package
content digest so active formal-gate packages can be excluded exactly. The raw
archives retain display names and full setup decks, but not package-code
digests; no digest may be guessed from a display name or deck.

Do not run the legacy whole-corpus Python-list loader against all ten days: it
can recreate the prior host-memory overload. Until the oracle index and pinned
package registry exist, iteration 15 may create only the immutable activation
receipt; fitting must fail closed and production must continue on the untouched
base learner.

## Required promotion gates

1. Exact boundary receipt and frozen-parent identity pass.
2. Oracle sidecar, active-gate exclusion, all-seven coverage, and bounded-memory
   staging pass.
3. Adapter-only fit, crash/resume parity, per-route validation, base bit-equality,
   and complete dormant merge pass.
4. Runtime recognizer evaluation passes independently for each route intended
   for activation.
5. Native-game evaluation and a separate explicit promotion decision pass.

Until all relevant gates pass, rollback is the untouched iteration-15 learner
with `matchup_adapters_enabled=false`.
