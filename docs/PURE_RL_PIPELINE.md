# Pure-RL Pipeline: Schema-v5 Production Runbook

This document is the source of truth for the post-audit training line. Older
phase names, overnight commands, Hammer-specific plans, and one-epoch notes are
historical only. A production run is not authorized merely because a checkpoint
exists: code, representation, remotes, lineage, collection, training, promotion,
and release gates must all pass.

## Non-negotiable contract

- Use feature schema **v5**. Legacy or schema-mismatched checkpoints fail closed.
- Accept only a compact production policy below **2 million parameters**. The
  current schema-v5 seed is 1,935,507 parameters. The broader 3.5M code guard is
  an emergency ceiling, not production acceptance.
- Train one policy on Inzi. Elmo and Bert provide whole-game collection capacity;
  they are not independent trainers.
- Keep search off during this line. Collection is sampled policy play; evaluation
  and submission are greedy.
- Run each iteration serially:

  ```text
  collect with incumbent
      -> two fresh-data AWR passes
      -> immutable candidate
      -> candidate-vs-incumbent promotion gate
      -> publish selected incumbent
      -> reload and verify every local leaf and required remote
      -> next collection wave
  ```

- Never overlap collection for iteration `t+1` with training or promotion for
  iteration `t`. That would mix policy versions and invalidate the shard lineage.
- Require all configured remotes in production and disable silent local fallback.
  A missing endpoint, stale digest, failed reload, failed pin, semantically failed
  remote job, or zero remote completions stops the wave.
- Keep auto-progress and submission **off by default**. A human-reviewed gate must
  name the exact accepted digest before either is armed.
- Do not hard-code Hammer as the specialist. Select a specialist by expected
  value under a versioned ladder mix after the core gate passes.

## Schema-v5 representation

Schema v5 binds each selectable option to its exact engine identity. The option
decoder receives a composite `(role, owner, area, index)` token, not independent
additive marginals:

- roles: source, target, tool, and energy;
- owner: acting seat, opponent, or unspecified;
- area: the engine area, with an explicit unknown value;
- index: the exact engine index, with an explicit unknown value.

The composite vocabulary has 10,140 rows (`4 x 3 x 13 x 65`). This preserves
pairing information for ordered and multi-select actions that would otherwise
collapse to the same owner/area/index marginals. Spatial slot embeddings preserve
board position as well. The legal-action enumerator must expose complete support;
it fails rather than silently truncating an oversized action space.

History is part of the policy input. When the temporal window rolls over, the KV
cache is recomputed from the retained raw window so online inference matches
offline training for both RoPE and learned positional modes. A checkpoint is
loadable only when its feature schema, model profile, decision context, and
trusted pure-RL provenance agree with the runtime.

## Learner contract

The learner uses advantage-weighted regression on the actions actually played,
plus a terminal-return value target:

- undiscounted terminal return (`gamma = 1`);
- a frozen, detached critic baseline for the iteration;
- raw and normalized advantages recorded separately;
- effective sample size and weight clipping recorded;
- fresh replay only, using the configured short shard window;
- optimizer, scaler, and global step restored on resume;
- incompatible optimizer state fails closed;
- no behavior-policy cross-entropy, starter-policy bootstrap, strategy-head loss,
  or MCTS visit target.

The production default is **two AWR passes per iteration**. Two passes are a
measured local default, not a teaching from Kaggle and not a permanent constant.
Change it only as a new, explicitly fingerprinted experiment after comparing
learning signal, effective sample size, overfit indicators, and held-out results.

## What the Kaggle evidence does and does not say

The field evidence supports compact models, fast simulation, large self-play
volume, refined curricula, a rich decision-complete state, replay inspection,
and early focus on a useful card subset. Relevant Abhyuday material:

- [Discussion 717697](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717697)
- [Discussion 724362](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/724362)
- [Discussion 723576](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/723576)
- [Discussion 709160](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/709160)

Taken together, these posts motivate the sub-2M profile, throughput work,
millions-of-games mindset, curriculum design, top-card coverage, and replay/state
audits. They **do not prescribe AWR, two epochs/passes, exact temperatures, gate
thresholds, replay-window length, or our promotion protocol**. Those are local
engineering hypotheses and must be measured as such.

Two additional public snapshots are useful for roster and meta cross-checks, not
as unreviewed training recipes:

- [beicicc: public experiment snapshot, Jul 15](https://www.kaggle.com/code/beicicc/ptcg-public-experiment-snapshot-jul15)
- [makimakiai: public 28+ / sample-4 roster update](https://www.kaggle.com/code/makimakiai/ptcg-public-28-plus-sample-4-roster-update)

## Ladder mix and specialist choice

The official [2026-07-12 episode dataset](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-12)
gives this observed top-ladder episode spread:

| Rank | Deck family | Episode prevalence |
|---:|---|---:|
| 1 | Alakazam | 35.7% |
| 2 | Crustle | 23.3% |
| 3 | Grimmsnarl | 9.4% |
| 4 | Garchomp | 5.6% |
| 5 | Cornerstone Ogerpon | 4.9% |
| 6 | Rocket Mewtwo | 4.5% |
| 7 | Starmie | 4.0% |
| 8 | Hammer | 3.8% |

These percentages measure **prevalence in that dated episode sample**, not deck
strength, matchup win rate, player skill, or future ladder share. Do not call
Alakazam the strongest deck from this table, and do not infer that Hammer is the
best specialist because an older plan named it. The table covers 91.2% of all
seat appearances in that sample; it is intentionally not renormalized to 100%.

Use a versioned ladder-mix artifact with its content digest in the manifest and
ledger. Derive deterministic collection quotas from that artifact, while keeping
promotion and official held-out decks fixed and excluded from training. Record
deck identity in every shard. A duplicate/missing deck, invalid distribution, or
mix-digest drift on resume is a lineage error.

After the core gate, estimate each specialist candidate's ladder value:

```text
expected_ladder_value(specialist)
    = sum(meta_weight[opponent] * gated_matchup_value[specialist, opponent])
```

Choose only among candidates with adequate, seat-balanced matchup coverage and
confidence bounds. Version the meta weights and evaluation matrix. Prevalence can
set evaluation priority; it cannot replace measured matchup value.

## Immutable lineage and iteration artifacts

Every production run has one append-only lineage. At creation, persist and hash:

- base checkpoint content and trusted schema-v5 provenance;
- actual model profile and parameter count;
- git revision plus source-tree content digest;
- full resolved training and collection configuration;
- deck-distribution artifact and digest;
- collection and held-out opponent identities and checkpoint digests;
- required remote endpoints;
- RNG seed and iteration number.

Resume only from the committed loop ledger. The current design fingerprint must
match the ledger and manifest. Never infer progress from a filename, directory
listing, or “latest checkpoint.” Partial shards and candidates are either
recovered under their recorded transaction or quarantined; they are never reused
as if committed. Smoke runs live in a separate namespace and cannot occupy a
production run name.

Each iteration writes immutable, digest-addressed artifacts for:

1. collection jobs and retained trajectories, including policy and opponent
   digests, seats, deck IDs, result status, and execution origin;
2. raw/normalized advantage metrics, ESS, losses, optimizer state, and two-pass
   completion;
3. the candidate checkpoint and its parent digest;
4. seat-balanced candidate-vs-incumbent promotion evidence and confidence result;
5. selected incumbent digest—candidate on pass, previous incumbent on reject;
6. successful local-leaf and required-remote reload/pin acknowledgements for that
   exact selected digest;
7. one terminal iteration commit.

No next-iteration job may be issued before item 7.

## Gates

### Collection gate

- Every trajectory identifies the exact incumbent, opponent, seats, and deck.
- The usable trajectory fraction meets the configured minimum.
- Required remote work actually completed remotely; fallback is zero.
- Engine failures, forfeits, timeouts, and semantically failed jobs are explicit
  metrics, not wins and not silently discarded.

### Candidate promotion gate

- Evaluate candidate versus incumbent with both seats and fixed decks.
- Use the configured bootstrap confidence bound and non-regression floor.
- Publish the candidate only on a pass. A rejection republishes the incumbent;
  it does not relabel the candidate as accepted.
- Reload the selected digest on every local leaf and required remote, then verify
  reported digest/version before collection resumes.

### Core held-out gate

- Keep official held-out opponents completely outside replay and curriculum data.
- Use at least the configured seat-balanced game count.
- Require both the aggregate point-win-rate threshold and the per-opponent floor.
- Exclude infrastructure failures from the competitive score while separately
  failing the reliability gate if those failures exceed tolerance.
- Treat repeated evaluation against the same public set as development feedback;
  reserve a locked release suite for final specialist/submission choice.

### Specialist and release gate

- Select the specialist by expected ladder value, never by a hard-coded deck name.
- Re-run core non-regression, the versioned ladder matrix, and the locked release
  suite on the exact proposed digest.
- Submission tooling must receive that exact accepted digest. It must never fall
  back to newest-by-mtime or a rejected candidate.

## Required-remotes policy

Production uses Inzi for training and local leaves, with Elmo and Bert as required
whole-game farms. Before launch:

- deploy one coherent code snapshot and the exact schema-v5 checkpoint;
- verify each service reports healthy leaves and the expected checkpoint digest;
- run a two-checkpoint canary so incumbent/opponent staging cannot alias;
- test reload and pin acknowledgement for the exact digest;
- complete at least one real trajectory on each remote;
- verify reconnect/retry still fails closed when a required endpoint is removed.

Set `POKEBOT_REMOTE_REQUIRE_ALL=1` and
`POKEBOT_REMOTE_NO_LOCAL_FALLBACK=1`. Do not weaken those settings to keep an
overnight job alive; stop, repair, repeat the canary, and then resume from the
ledger.

## Safe preflight and launch template

Do not copy this template into production until the current code tests, coherent
remote deployment, two-checkpoint canary, and real per-remote canary have all
passed. Fill every placeholder deliberately. Use a new immutable run name.

```bash
# Supply these values deliberately. `${name:?message}` aborts before launch when
# any required value is missing; no production path or run name is inferred.
: "${POKEBOT_PYTHON:?set the absolute training-environment Python path}"
: "${RUN_NAME:?set a new immutable run name}"
: "${BASE_CHECKPOINT:?set the verified schema-v5 seed path}"
: "${REMOTE_ENDPOINTS:?set both required host:port endpoints}"
: "${ITERATIONS:?set the reviewed iteration count}"
: "${GAMES_PER_ITER:?set the reviewed games-per-iteration count}"
: "${HELDOUT_GAMES:?set the reviewed held-out game count}"

# Wiring-only local canary. It writes to the smoke namespace.
POKEBOT_PYTHON="$POKEBOT_PYTHON" bash scripts/canary_pure_rl.sh

# Required production controls.
export POKEBOT_PYTHON
export POKEBOT_REMOTE_REQUIRE_ALL=1
export POKEBOT_REMOTE_NO_LOCAL_FALLBACK=1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0

# Template only: quick preflight stays enabled; automation stays explicitly off.
"$POKEBOT_PYTHON" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile quick \
  --no-auto-progress \
  --remote-worker-endpoints "$REMOTE_ENDPOINTS" \
  --log "outputs/logs/${RUN_NAME}.log" \
  -- \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --iterations "$ITERATIONS" \
  --games-per-iter "$GAMES_PER_ITER" \
  --train-epochs 2 \
  --heldout-games "$HELDOUT_GAMES"
```

Do not use `--preflight-profile none` for a new production lineage. Do not pass
`--auto-progress`, and do not start submission tooling, until the exact core gate
artifact and chosen specialist have been reviewed.

## Stop conditions

Stop the run rather than improvising if any of these occur:

- schema, model-profile, source, config, deck-mix, opponent, or parent-digest
  disagreement;
- a required remote is absent, returns a stale digest/version, or completes no
  assigned work;
- silent local fallback or mixed execution origin;
- partial/duplicate iteration commit or an unowned artifact;
- too few usable trajectories, collapsed effective sample size, non-finite loss,
  or incompatible optimizer state;
- candidate promotion, per-opponent held-out, core, specialist, or locked release
  gate failure;
- any tool attempts to select “latest” instead of an exact accepted digest.

Quarantine the affected lineage without deleting it, record the reason, repair
the cause, repeat the relevant canary, and resume only when the immutable ledger
permits it.
