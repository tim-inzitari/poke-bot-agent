# Codex account handoff: legacy RTP retirement and r207 MCTS/BO1000

Snapshot date: 2026-08-10 (America/New_York)

This is a non-authoritative transfer note for another Codex account. The next
controller must read `AGENTS.md` and all of `GOAL.md` first, then the typed
sources named below. Do not treat this note as a replacement for those files.
The working tree is intentionally very dirty and contains concurrent user work;
never reset, clean, checkout, delete, or rewrite unrelated files.

## Owner boundary

Legacy recursive RTP is fully abandoned under revision 210. The separate
revision-202/205/207 simulator-backed MCTS experiment is explicitly non-RTP and
may continue offline, but it has no production action or serving authority.

Canonical files observed at this handoff:

- `GOAL.md`: revision 211, SHA-256
  `725a3e9c7783a5c85b88b042234d75a534f410932531d0d1d2bbe61ab1bd6190`.
  Revision 211 is unrelated Replay Inspector work; do not roll it back while
  preserving revision 210.
- `state/alakazam-rtp-abandonment-r210.json`: SHA-256
  `bb9eaa02398175fc5c9bd8e29ce290f102afff234b6d27bf7588fc1e53f09961`.
- `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json`: SHA-256
  `d9cb5f8d15e2bebbcbf943f5a273a4116703c3e8549a3328b7d78d161f7b5dce`.

Rehash and reread these files before acting because another authorized session
may advance `GOAL.md` after this snapshot.

## Legacy RTP retirement: completed and independently audited

Managed user service on `inzi@192.168.1.151`:

- Unit: `pokebot-alakazam-rtp-r198-three-arm-eval.service`
- Final tuple: loaded/failed, `MainPID=0`, `ExecMainPID=1431670`,
  `Result=signal`, `ExecMainCode=2`, `ExecMainStatus=15`, `NRestarts=0`,
  `Restart=no`, invocation
  `1aaed5eeae4349d78c27a29d80bd2441`.
- The signal result is the owner-authorized managed stop, not an evaluator
  failure.
- No service job was queued at the final audit.

Persistent retirement guard:

- Remote path:
  `/home/inzi/.config/systemd/user/pokebot-alakazam-rtp-r198-three-arm-eval.service.d/99-r210-retired.conf`
- Local source:
  `deploy/systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service.d/99-r210-retired.conf`
- Mode/size/hash: `0444`, 186 bytes,
  `2dda98aeee06b3488bf32ade03aaca06f1fcd7a4de8e3d7617425fca55d5f24c`.
- Loaded semantics: `RefuseManualStart=yes` and
  `ExecCondition=/usr/bin/false`.
- `NeedDaemonReload=no`, `UnitFileState=linked`; no wants, trigger, reverse
  dependency, mask, or queued job was found.

The exact canonical linked unit remains preserved:

- Fragment target:
  `/home/inzi/poke-bot-agent-deployments/alakazam-rtp-r198-three-arm-eval-src-36a15c52a4c7/systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service`
- Unit SHA-256:
  `36133a5280a87184c35a754595e8bc7e2feffe7d6099cc27b751cd79ce7b8eb5`.
- Source manifest SHA-256:
  `a4e20e8836702ab5abae18b7dcf3f4c1c7f1336d60f172583a0dca6ceb9c41a1`.

Retirement receipt:

- Local: `state/alakazam-rtp-abandonment-retirement-guard-r210.json`
- Remote:
  `/home/inzi/poke-bot-agent/outputs/state/alakazam-rtp-abandonment-retirement-guard-r210.json`
- Raw SHA-256:
  `d0ee2255bf2b5e4abd2c1b9eaaff39343997c2452578d53347927fa5b2f75db0`.
- Canonical payload digest:
  `sha256:18852cd7726044da2de19bc5469a463f4af49d4153bdebf1b2b03ea0ebc6666c`.

Attempt-10 preservation boundary:

- 761 transcripts and 761 receipts, all completed leaf files mode `0444`.
- 253 complete three-arm cells, plus NO-RTP and direct arms of cell 253.
- Arm counts: NO-RTP 254, direct 254, recursive 253.
- Recursive cell 253 is unscored and has only its immutable fence.
- No failed-worker evidence, terminal evaluator result, compiler result, HOLD,
  promotion receipt, or stage receipt exists.
- Completed-evidence snapshot digest:
  `sha256:a561ed820d00b8b9460c0ea0d9aa17c8e0fa82c7834451e7bb13c370b742628b`.
- Eval-tree content/mode snapshot digest:
  `sha256:6868ec957f9dd266c18e5d40f2f09fef391dce67a67864e4869be848a3e39ad7`.
- Roots are still mode `0755`; use only the exact snapshot digests above and
  never claim terminal sealing or a completed efficacy result.

Never start, restart, reset-failed, enable, disable, mask, unmask, unlink, or
probe-start this legacy service. Never retry attempt 10, create another legacy
RTP evaluation, use its sidecar/executor, train on the partial rows, or delete
its evidence.

## Separate r207 MCTS core: implemented and reviewed, not launch-ready

Reviewed full-turn/inter-turn search core:

- `poke_bot/recursive_turn_planner/simulator_one_turn_expectimax.py`
  SHA-256
  `07aa0ecac4893502742c7c5f5da7198aaf585d2282a1234193c4342bc5967c54`.
- `tests/test_simulator_one_turn_expectimax.py`
  SHA-256
  `acf65a33eeb04cb75221c171126b8dd53d44e20e92dec89fcd462cfe6ee4a8b1`.

Public integration API:

- `PolicyDecisionView` / `PolicyDecisionFactory`
- `R207ArenaAdapter`
- `MCTSExpansionProfile`
- `SimulatorInterTurnMCTSSession.capture_and_plan()`
- `SimulatorInterTurnMCTSSession.observe_real_action()`
- `SimulatorInterTurnMCTSSession.plan_next()`
- `MCTSTurnTelemetry.to_bo1000_turn_telemetry()`

Verified behavior:

- Searches multiple same-seat atomic decisions but returns only one real action.
- Uses the canonical typed 20-second actual-turn / 5-second action clocks.
- Exact direct-policy action is mandatory and is the fail-closed fallback.
- Exact finite chance retains every outcome child and recomputes rational
  weighted backups on later simulations.
- Opponent/private/unresolved information remains a boundary.
- Cached deterministic chains require explicit realized deterministic
  attestation at every hop; omitted, chance, opponent, or fingerprint-mismatch
  transitions rebuild.
- Root selection, tree digest, cache installation, score creation, and
  telemetry are all deadline charged.
- It imports no legacy RTP sidecar/executor and has no service/action authority.

Related component hashes:

- `r207_simulator_arena.py`:
  `06b4e1abe90732861dc31d3118c24d70344f90757867357a99e8ba592f92fd41`
- `neural_leaf_reranker.py`:
  `636671eb69f466f560175eb8c199b72ab41e3f844f74a5552ac19ab2f559504b`
- `r207_frozen_leaf_calibration.py`:
  `e0b846d662b3af9929bf6c7b1ffee5540bc056c290e1b2c23cd4021113133294`
- `bo1000_evaluation.py`:
  `4c71b28d3491d9d92233ca9c14c158d79e7f1112d7a855d02c28907d5aeaab2f`
- `bo1000_pair_runner.py`:
  `89db9ba780f468b40fa86712d1750a0ea4ee2ac7731631ea9c6ab4e677fa536c`
- `bo1000_remote_pair_protocol.py`:
  `3c63f9fb5ff0ea7fa0ae1db3efed41a6f23e9cbb12f51c473906b7f1f8fd0eaf`

The current native V3 foundation is deliberately not a working successor
engine. Its C++ `expand` path returns `AUDIT_REQUIRED` and does not step the
engine. Do not relabel it as launch-ready:

- `engine_patches/r207_v3/RtpPlannerSuccessorArenaV3.cpp`:
  `ca002b5ad03963193c0a358c2ad62845750a83115dcb9859170fe68634beec9b`
- `engine_patches/r207_v3/patchset.json`:
  `2addd9e32488d015fe5858a6ec60578fa12df59d1d45944c91c779a7ca0a88f0`
- `poke_bot/engine_rebuild/rtp_planner_successor_v3.py`:
  `20e2eb36804254d09546767ae2162e591b9458522eada153c906ab0f628709c5`
- `poke_bot/engine_rebuild/rtp_planner_successor_v3_build.py`:
  `c778904a1aefe60ee4e696f34f806e0a7beff09234753cd4dde57970372dc213`

## BO1000 launch blockers

No BO1000 was launched. Launch remains fail-closed until all of these exist:

1. A real information-set-safe native V3 `SuccessorArena` and factory with
   arbitrary-decision opaque capture/clone/step, exact future legality,
   terminal parity, dynamic hidden/random provenance, and exact finite-chance
   receipts. Static card whitelists or guessed hidden zones are insufficient.
2. A real checksum-bound r195 frozen-model reranker plus an eligible sealed,
   source-excluded nonterminal calibration/no-training receipt. The compiler
   exists, but no eligible real receipt has been issued.
3. Session-to-game/pair-runner integration, local deterministic canary,
   compiler preflight, remote pair job deployment, local/remote determinism
   receipt, per-host noninterference receipts, and a new content-addressed
   source/evaluation/output identity.

Frozen comparator identity remains submission `55378392`, checkpoint
`sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a`,
bundle
`sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145`.
The BO1000 is exactly 500 RNG-matched seat-swapped pairs / 1,000 games, with
MCTS in each seat exactly 500 times and no early stop.

Historical capacity observations are not launch receipts: Elmo and Bert were
idle but exposed no r207 pair/search job kind and pinned a different checkpoint;
train had no managed r207 worker. Reprobe every host. Do not reuse per-turn or
per-simulation LAN RPC; dispatch one immutable pair envelope per host and keep
simulation/search/leaf inference host-local.

## Validation at handoff

The final retirement + r202/r205/r207/BO focused matrix passed:

```text
168 passed in 2.23s
```

The MCTS core and the related handoff modules passed targeted Ruff and
`py_compile`. A broader Ruff invocation that included
`r207_simulator_arena.py` still reported eight style findings in that file;
do not claim repository-wide lint clean until those are intentionally reviewed.

Re-run the exact test command from `CODEX_ACCOUNT_BOOTSTRAP_PROMPT_R210_R207.md`
before changing code.

## Suggested agent split for the receiving account

Use independent agents only when authorized by that account's instructions:

1. **Native successor agent** — implement and receipt dynamic hidden/random
   provenance plus safe opaque V3 capture/clone/step/observe/finite-chance.
2. **Frozen-model evidence agent** — produce source-excluded heldout
   nonterminal predictions and compile the real calibration/no-training receipt.
3. **BO1000 integration agent** — wire the reviewed session into fresh-process
   game/pair runners and the durable whole-pair remote protocol.
4. **Release auditor** — independently verify identities, clocks, pairing,
   host noninterference, compiler semantics, and content-addressed publication.

The receiving primary agent should integrate only after each preceding artifact
is immutable and independently verified. Do not launch merely because the
Python search core is green.
