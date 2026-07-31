# Cursor takeover handoff

Updated: 2026-07-31  
Repository: `/Users/tsinzitari/Documents/poke-agent-codex`  
Current owner contract: `GOAL.md`, revision 89  
Copy-paste task prompt: `CURSOR_PROMPT.md`

## Read this first

1. Read `AGENTS.md` and `GOAL.md` completely before acting.
2. Then read the canonical sources named by `GOAL.md` for the action being taken. `ops/current_goal_requirements.json` is only a compatibility projection and never overrides `GOAL.md`.
3. Preserve the dirty worktree. At this handoff it contains 104 tracked changes and 110 untracked files. Do not run `git reset`, `git clean`, `git checkout --`, broad formatters, or mass rewrites.
4. Never terminate, signal, disconnect, replace, or classify an SSH, Codex, Cursor, terminal, editor, or other interactive session as stale. The owner LAN identity in `AGENTS.md` is hard-allowed.
5. Operate training and dashboard workloads only through their declared service manager. Never use `kill`, `pkill`, `killall`, shell signals, or process-tree termination.
6. Inspect live state before changing anything. Do not restart healthy training merely to reconcile planning metadata.

## Current objective

The active specialist is Alakazam, training the real H10 policy by ordinary same-archetype pure RL from the accepted Gen9 policy. The attempted direct historical H10 migration failed closed and must remain preserved as immutable evidence.

Alakazam's exact iteration contract is:

- exactly 16,384 games per iteration;
- exactly one learner epoch;
- at most 189 iterations;
- exact 8,192/8,192 seat balance at assigned, retained, and consumed stages;
- exactly 2,048 self-play games plus 14,336 public/specialist games;
- the current public plan is 8,168 strong-public plus 6,168 diverse-public games;
- three independent terminal gates: public pool win rate at least 0.58, active archetype win rate at least 0.62, and live community metagame win rate at least 0.58;
- stop at the first checkpoint that passes all three gates, then freeze/register/submit and advance to Marnie;
- population training remains blocked until every canonical ladder family has a passing frozen representative, except Slowking is the one explicitly recorded failed experiment.

Canonical receipts:

- `state/final_format_alakazam_h10_iteration_size_r81.json`
- `state/final_format_alakazam_h10_mix_r82.json`
- `state/final_format_alakazam_h10_bert_mps_isolation_r86.json`

## Active recovery state from revision 89

Iteration 0 collection already completed. The first learner attempt failed during the adapter phase with a CUDA OOM after the ordinary epoch completed. Revision 89 launched a recovery that reuses that complete corpus; recollection is forbidden.

Required learner bounds:

- game chunk: 240 games;
- ordinary decision batch: at most 4,096;
- warmup decision batch: at most 4,096;
- adapter decision batch: at most 2,048;
- release CUDA cache before adapter fitting;
- recursively split only the failing learner batch on OOM;
- never recollect the completed iteration-0 corpus.

Last receipt-backed active registry:

```text
/home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79/runtime/specialist_runtime_registry_h10_r89_iter0_elmo_only_batch4096_adapter2048.json
sha256:661450f4e08f6e1f8bea5184ad1fea0f4bb3b85d34f8d04550591277918e1bf2
```

Revision 89 recorded PID `706275`, restart count `0`, and Elmo as the only configured remote. Treat those values as a receipt snapshot, not as present-time truth. Audit the managed service and immutable artifacts before acting.

The Blackwell service is:

```text
pokebot-final-format-alakazam-r79-h10.service
```

The current service unit references the revision-89 registry in its preflight, gate handler, and `ExecStart`. Expected run directory:

```text
/home/inzi/poke-bot-agent/outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k
```

Important evidence:

```text
/home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79/logs/h10_launcher.log
/home/inzi/poke-bot-agent/outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k/commits/iter_00000.json
```

First determine whether `iter_00000.json` exists and validates. If the service is healthy and iteration 0 is still progressing, leave it alone and supervise it. If it is inactive, diagnose from the managed-service status, journal, loop state, and receipts before any service-manager action. Do not recollect iteration 0.

## Bert status and required next boundary

Bert was intentionally removed from production for the remainder of iteration 0. The production LaunchAgent is unloaded and ports 8766 and 8776 had no listener at this handoff. Do not restore Bert before the iteration-0 commit validates.

The optimized exact-H10 MPS benchmark that was still described as in progress by the compatibility projection has now completed locally:

```text
/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/outputs/benchmarks/h10_r84_mps_cache0_leaf4_512_45g_status.json
file sha256:3e063202c58d021f60c383e15b30bc5ab08181effe06f393f11291258240c74c
checkpoint sha256:e65123a13abb61332fe89e66946103a83c766e2f15315b945bfe9b6bf0c2d32e
```

Recorded result: complete, 512/512 games, zero errors, usable fraction 1.0, 0.221015 games/s, 25.404647 decisions/s, 21.411 GiB process-tree RSS, 23.678 GiB host-free RAM, production inactive. The result recommends comparison with the existing CPU baseline; it does not itself select MPS.

Existing short exact-H10 whole-game comparison:

```text
/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/outputs/benchmarks/h10_r82_whole_game_status.json
```

That receipt has 16/16 valid CPU games at 0.420205 games/s and 49.689188 decisions/s versus an older MPS result at 0.136675 games/s. It recommends `cpu-2t`. Because the optimized MPS result is 512 games while the CPU comparison is only 16 games, do not silently infer that this is the final canonical selector. Validate whether the current benchmark contract permits this comparison. If it does not, run an isolated, managed, matching 512-game CPU benchmark with the same checkpoint, seeds, game contract, and production-equivalent topology. Emit a checksum-bound comparison/selection receipt.

Only after iteration 0 commits and the benchmark comparison is canonically valid may Bert return at a safe boundary. Restore it through launchd, use the exact H10 checkpoint above, enforce the 45 GiB process-tree guard and 24 GiB host-free minimum, select the fastest valid backend, verify `/health`, and update only the runtime registry for future work. Do not rewrite the immutable revision-89 iteration-0 registry or claim Bert participated in iteration 0.

## Elmo and Blackwell contracts

- Elmo is the sole revision-89 remote for iteration 0 and may assist all rollout/evaluation phases unless a later canonical receipt changes that.
- Elmo H10 profile: 36 workers, four inference leaves, four environments per worker, 45 GiB process-tree guard, 24 GiB host-free minimum, 64 GiB no-swap host limit.
- Elmo must use checkpoint `sha256:e65123a13abb61332fe89e66946103a83c766e2f15315b945bfe9b6bf0c2d32e`.
- Blackwell exact profile: 96 workers, 96 in flight, four environments per worker, four self-play learner leaves, twelve gate leaves, one GPU shared by inference and the sole active learner.
- Keep `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`.
- Only the active learner may create gradients. Remote helpers are inference/evaluation only.

## Staged optimization after the safe boundary

Revision 87 stages value-only frozen-baseline work and one-batch CPU prefetch for activation at the next safe learner boundary, after exact Blackwell parity is proved. Ten focused AWR tests passed on the CPU exact path. Packed temporal batching remains disabled because last-bit exactness failed; do not enable it without a new exact-parity receipt.

## Prior result that must remain preserved

Slowking failed its corrected iteration-5 gate and has a sealed 8,192-game iteration-6 corpus with 617,296 decisions. It was not frozen, registered, submitted, or served. The canonical registry therefore contains 14 frozen specialists plus one explicit failed-experiment exception. Do not reinterpret the sealed corpus as a pass.

## Repository state

- Branch: `codex/worksession-20260728`.
- The branch was 16 commits ahead of its upstream at the previous audit.
- At this update: 104 tracked changed files and 110 untracked files.
- Tracked diff: about 10,957 insertions and 1,032 deletions.
- These changes are active user/agent work. Preserve unrelated and overlapping edits.
- `CURSOR_HANDOFF.md` and `CURSOR_PROMPT.md` are handoff artifacts; do not mistake them for canonical runtime receipts.

## Definition of a successful takeover

Cursor should produce evidence for each completed boundary:

1. managed-service state, restart count, main PID, and current phase;
2. exact runtime-registry path and digest;
3. validation of the existing iteration-0 corpus and the iteration-0 commit/checkpoint, explicitly confirming no recollection;
4. checksum-bound Bert benchmark comparison and backend-selection receipt;
5. post-commit Bert restoration receipt, health evidence, and future-iteration registry mapping, if and only if eligible;
6. exact-parity evidence before activating the revision-87 learner optimization;
7. focused tests for any code/config changes;
8. an update to compatibility projections only after runtime truth is established, without rewriting immutable history.

Do not merely rewrite dashboards or prose to look current. Runtime truth comes from `GOAL.md`, the canonical selector, immutable receipts, and managed-service state.
