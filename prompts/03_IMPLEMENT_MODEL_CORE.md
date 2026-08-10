# Prompt 3 — Action Decoder, Heads, Dynamics, and Root Planner

```text
Read AGENTS.md, the completed repository mapping, docs/poke_rlm/01_ARCHITECTURE.md, docs/poke_rlm/03_PARAMETER_BUDGET.md, and Phases 2–4 of the roadmap.

Implement the smallest repository-native model path that supports:
1. one shared state encoding;
2. one batched decoder over all current legal actions;
3. fused policy, distributional Q, multi-horizon value, successor, state-value, and uncertainty outputs;
4. short action/macro-conditioned latent dynamics;
5. a finite typed root-plan proposer producing 4–8 candidates;
6. compilation into the typed plan IR and exact static validation.

Start with the pilot-compatible width if checkpoint compatibility requires it. Keep configuration capable of the recommended base_384 profile. Do not implement recursive depth yet unless Phase 4 tests require a stub interface.

Hard requirements:
- no full-backbone call per legal action;
- no Python loop over legal actions in the deployed forward path;
- all tensor shapes documented and asserted;
- current checkpoints remain loadable through an explicit migration/compatibility path;
- current policy remains the default action selector;
- new outputs run in shadow mode first;
- exact parameter count by module is printed and stored;
- p50/p95/max latency and peak memory are measured on representative action-set sizes.

Add unit, mask/padding, checkpoint, deterministic-forward, maximum-action, and static-plan-validation tests. Run tests and benchmarks. Do not start large training; provide exact pilot training commands/configs after the implementation passes smoke gates.
```
