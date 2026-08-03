# Prompt to paste into Cursor

Take over this repository and continue the active Pokemon RL goal autonomously and conservatively:

`/Users/tsinzitari/Documents/poke-agent-codex`

Before doing anything, read `AGENTS.md`, `GOAL.md` revision 89, and `CURSOR_HANDOFF.md` completely. Then read every canonical source named by `GOAL.md` that is relevant to the action you are about to take. Treat `GOAL.md` as authoritative; `ops/current_goal_requirements.json` is only a compatibility projection.

Preserve the worktree exactly. The large controller/runtime change set is already committed as `db5485a`; the two current tracked modifications are `CURSOR_HANDOFF.md` and `CURSOR_PROMPT.md`. Do not run `git reset`, `git clean`, `git checkout --`, broad formatters, or mass rewrites. Do not overwrite concurrent or unrelated edits.

Safety is non-negotiable: never terminate, signal, disconnect, replace, or classify an SSH, Codex, Cursor, terminal, editor, or other interactive session as stale. Never use `kill`, `pkill`, `killall`, shell signals, or process-tree termination. The LAN identity in `AGENTS.md` is owner-controlled and hard-allowed. Control training/dashboard workloads only through their declared service manager, and never restart healthy training to make metadata agree.

Start with a read-only audit and report the evidence briefly, then continue safe in-scope work without waiting for permission:

1. Inspect launchd state for both `com.pokebot.remote-worker-8766-h10-r80` and `com.pokebot.bert-post-alakazam-iter0-rejoin-r89`, plus listeners on ports 8766/8776. The production worker should remain unloaded for iteration 0. The polling rejoin job currently runs every 15 seconds, its marker is absent, and its script hard-codes MPS before the required CPU/MPS selector exists. If the canonical selector receipt is still absent, immediately hold the poller using `launchctl bootout gui/$(id -u)/com.pokebot.bert-post-alakazam-iter0-rejoin-r89`. This is a managed-workload safety hold, not permission to signal any process. Patch/replace its workflow so it consumes a checksum-bound backend selector, and re-bootstrap it only after the post-commit migration is eligible.

2. On Blackwell, inspect `pokebot-final-format-alakazam-r79-h10.service` through `systemctl --user show/status` and `journalctl --user -u`. Record active state, substate, result, main PID, restart count, invocation ID, and current phase. At handoff it was healthy and active with MainPID 706275, restart count 0, invocation `e178fddb48844a2ba27f2ce429b363e6`, training PID 706419 in `rl-agreement parent`, zero monitor OOMs, and no iteration-0 commit. Recheck rather than assuming those volatile values. Validate the exact revision-89 registry and its SHA-256:

   `/home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79/runtime/specialist_runtime_registry_h10_r89_iter0_elmo_only_batch4096_adapter2048.json`

   Expected digest:

   `sha256:661450f4e08f6e1f8bea5184ad1fea0f4bb3b85d34f8d04550591277918e1bf2`

3. Inspect the run directory and immutable evidence:

   `/home/inzi/poke-bot-agent/outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k`

   Check loop state, the launcher log, the completed iteration-0 collection, and `commits/iter_00000.json`. Iteration 0 must be recovered from the already-complete corpus with ordinary/warmup batches capped at 4,096 decisions and adapter batches capped at 2,048. CUDA cache release before adapter fitting and recursive OOM splitting are required. Recollection is forbidden.

4. If iteration 0 is still running and healthy, do not restart or perturb it; supervise it until the commit validates. If the service is inactive or failed, diagnose using the managed-service state, journal, loop state, and immutable receipts. Use only the declared service manager for a justified recovery and preserve the completed corpus.

5. Bert must stay out of production for the rest of iteration 0. Locally validate this completed optimized MPS benchmark and bind your findings to its checksums:

   `/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/outputs/benchmarks/h10_r84_mps_cache0_leaf4_512_45g_status.json`

   Expected file SHA-256:

   `3e063202c58d021f60c383e15b30bc5ab08181effe06f393f11291258240c74c`

   Expected H10 checkpoint:

   `sha256:e65123a13abb61332fe89e66946103a83c766e2f15315b945bfe9b6bf0c2d32e`

   It completed 512/512 MPS games with zero errors at 0.221015 games/s and recommends comparison with the CPU baseline. Compare it with `h10_r82_whole_game_status.json`, whose 16-game CPU result is faster but shorter. Determine from the canonical benchmark contract whether that is sufficient for backend selection. If it is not sufficient, run an isolated managed 512-game CPU benchmark using the same checkpoint, seeds, exact-H10 game contract, and production-equivalent topology. Do not activate production while benchmarking. Create a checksum-bound comparison and selector receipt; choose the fastest valid backend, not a preferred device by assumption.

6. Restore Bert only after `iter_00000.json` exists and validates and the benchmark selector receipt is valid. Do not blindly use the staged revision-90 registry/service or the current rejoin script: they assume MPS. Regenerate or validate them against the selected backend. Restore Bert through launchd with the exact H10 checkpoint, 16 workers, four home-affine inference leaves, 45 GiB process-tree RSS guard, and 24 GiB host-free minimum. Use BF16 autocast/no per-batch cache eviction only if MPS wins the canonical comparison; otherwise use the selected CPU profile. Verify the production endpoint and `/health`. Update a new future-iteration runtime registry and compatibility projection. Never modify the immutable revision-89 iteration-0 registry or claim Bert helped collect iteration 0.

7. At the next safe learner boundary, validate exact Blackwell parity for revision 87's value-only frozen-baseline path and one-batch CPU prefetch before activation. Keep packed temporal batching disabled unless a new receipt proves last-bit exactness. Run focused tests for every code/config change.

8. Continue Alakazam under the exact contract: 16,384 games/iteration, one learner epoch, max 189 iterations, exact 8,192/8,192 seats, exactly 2,048 self-play plus 14,336 public/specialist games, and all three independent terminal gates. Stop at the first all-gates pass, then freeze/register/submit and advance to Marnie. Do not start population training yet.

For every material action, preserve or create immutable evidence: service snapshot, registry digest, corpus/commit validation, checkpoint digest, benchmark selector, restoration/health receipt, and focused test results. Update projections and dashboards only after runtime truth is established. If you encounter a checksum mismatch, missing canonical contract, ambiguity that would change the owner-defined design, or need authority for destructive/external action, stop and report the exact blocker instead of guessing.

Do the work; do not only return a plan. Leave healthy active training alone while you validate the surrounding state.
