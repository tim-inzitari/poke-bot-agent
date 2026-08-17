# Cursor phase-plan audit

Audited on 2026-07-16 from Inzi's latest Cursor plan and its earlier planning
transcript:

- `/home/pokebot/.cursor/plans/pure_rl_pipeline_a921bc49.plan.md`
- `/home/pokebot/.cursor/projects/home-inzi-poke-bot-agent/agent-transcripts/277e348e-9f73-4774-9d12-56f5d31ab048/277e348e-9f73-4774-9d12-56f5d31ab048.jsonl`

## Verdict

The plan is useful historical context, but it is not a safe executable
production specification. It mixes three incompatible phase taxonomies and
has no authoritative phase-state artifact. Its canary used `preflight none`,
did not prove that either LAN worker completed a real trajectory, and allowed
remote work to fall back locally. Its final specialist was fixed to
`hammer-pult` without evidence that Hammer maximized expected ladder value.

The quarantined run confirms the risk: only roughly 1,149–1,277 of 2,048 jobs
per affected iteration were valid, roughly 771–899 failed, held-out win rate
remained around 1.5–3.5%, and the advantage signal was effectively dead.
Remote connection failures were logged, but the run continued, so advertised
distributed execution was not established.

## Conflicting phase definitions

The historical five-stage graph was:

1. `core_bc`
2. `core_deep_search`
3. `core_gate`
4. `hammer_warmstart`
5. `hammer_search_rl`

Later Cursor notes used a different Stage A/E vocabulary, while the latest plan
described a core-to-specialist chain. None binds phase transitions to one
append-only state ledger, exact checkpoint digest, source digest, deck-mix
digest, and completed gate result. Phase E also fixed AWR to one epoch; that is
an implementation choice, not a conclusion supported by the Kaggle posts.

## Replacement production phases

1. Freeze source, schema, model profile, seed checkpoint, opponents, and ladder
   mix into an immutable manifest.
2. Run a real distributed canary: exact checkpoint reload plus a complete,
   valid trajectory on every required endpoint.
3. Collect a ladder-weighted, seat-balanced core curriculum. Keep training,
   promotion, recurring validation, and locked release populations distinct.
4. Train fresh-data AWR with a frozen per-iteration critic baseline. Use two
   measured passes by default; record effective sample size and stop changing
   this value by folklore.
5. Promote candidate versus its exact incumbent before publishing weights.
   Record candidate, parent, optimizer, and checkpoint digests append-only.
6. Select any specialist only by expected value against the versioned ladder
   distribution. Do not hard-code Hammer.
7. Run a locked release evaluation once, then package only the checkpoint named
   and hashed by the passing gate. Submission remains an explicit action.

Automatic core→specialist→submit progression is therefore off by default.
When explicitly armed, it requires a specialist archetype and exact gate
checkpoint digest; it never chooses `latest` weights.
