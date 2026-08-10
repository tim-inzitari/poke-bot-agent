# Paste this into the receiving Codex account

Work in `/Users/tsinzitari/Documents/poke-agent-codex` and continue the
separately versioned r207 simulator-backed MCTS/BO1000 work. First read
`AGENTS.md` and all of `GOAL.md`, then read:

1. `CODEX_ACCOUNT_HANDOFF_R210_R207.md`
2. `state/alakazam-rtp-abandonment-r210.json`
3. `state/alakazam-rtp-abandonment-retirement-guard-r210.json`
4. `state/alakazam-chance-aware-inter-turn-mcts-r202.json`
5. `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r205.json`
6. `state/alakazam-chance-aware-inter-turn-mcts-bo1000-r207.json`

The working tree is intentionally dirty with concurrent user work. Preserve all
existing edits and untracked artifacts. Never reset, clean, delete, or rewrite
unrelated files.

Legacy recursive RTP is permanently abandoned. Never start/restart/probe its
service, retry attempt 10, create another legacy RTP candidate/evaluation, use
its sidecar/executor, train on its partial rows, alter selectors, or delete its
evidence. Perform only the read-only r210 verification in the handoff if needed.

Continue the non-RTP r207 MCTS toward a real BO1000, but do not launch until the
typed prerequisites are immutable and valid. The reviewed Python search core is
green; the current native V3 foundation is not a simulator successor engine and
returns `AUDIT_REQUIRED`. The immediate critical path is:

1. Implement an information-set-safe native V3 opaque successor ABI with
   dynamic hidden/random provenance, exact future legality, terminal parity,
   and exact finite-chance receipts.
2. Bind that ABI to `SimulatorInterTurnMCTSSession` and pass receipt-backed
   integration tests.
3. Produce a real source-excluded frozen r195 nonterminal calibration/no-training
   receipt and attach the verified reranker.
4. Integrate fresh-process games into the pair runner, run local determinism and
   compiler canaries, then preflight Elmo, Bert, and train before any remote
   pair-envelope deployment.
5. Publish a new content-addressed evaluation identity and only then run all
   500 matched pairs / 1,000 games.

Before edits, re-run:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --isolated --no-project \
  --with pytest --with PyYAML --with torch \
  python -m pytest -q -p no:cacheprovider \
  tests/test_alakazam_legacy_recursive_rtp_abandonment_r210.py \
  tests/test_stage_alakazam_rtp_r198_three_arm_eval.py \
  tests/test_stage_alakazam_rtp_r198_source_snapshot.py \
  tests/test_alakazam_chance_aware_inter_turn_mcts_r202.py \
  tests/test_alakazam_chance_aware_inter_turn_mcts_bo1000_r205.py \
  tests/test_alakazam_chance_aware_inter_turn_mcts_bo1000_r207.py \
  tests/test_chance_aware_inter_turn_tree.py \
  tests/test_r205_neural_leaf_reranker.py \
  tests/test_r207_frozen_leaf_calibration.py \
  tests/test_r207_simulator_arena.py \
  tests/test_simulator_one_turn_expectimax.py \
  tests/test_bo1000_evaluation.py \
  tests/test_bo1000_pair_runner.py \
  tests/test_bo1000_remote_pair_protocol.py
```

Expected snapshot result: `168 passed`.

Do not claim BO1000 launch readiness or results until the real native successor,
calibration, pair integration, determinism, host, and publication receipts all
exist. Lead every update with the actual executable boundary, not scaffold
status.
