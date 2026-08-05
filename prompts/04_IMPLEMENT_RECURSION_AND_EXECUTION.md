# Prompt 4 — Bounded Recursion, Stateful Execution, and Repair

```text
Read AGENTS.md, docs/poke_rlm/01_ARCHITECTURE.md, docs/poke_rlm/02_TYPED_PLAN_IR.md, docs/poke_rlm/05_INFERENCE_AND_LATENCY.md, and Phases 5–6 of the roadmap.

Implement:
- one shared recursive planner cell reused at every depth;
- batched refinement of all subgoals at the same depth;
- depth 0, 1, and 2 modes; depth 3 only as an explicitly disabled audit option;
- complexity router choosing direct, root-only, or recursive paths;
- explicit stop and compute-cost outputs;
- stateful plan cursor across CABT micro-decisions;
- current legal-list selector resolution before every action;
- symbolic resource ledger updates;
- conditional branch evaluation;
- bounded repair triggers and deterministic fallback;
- a single shared whole-turn deadline/model/simulator/node/depth budget.

Enforce hard caps in code. Prove recursive parameter sharing by test. Do not let the planner emit natural language or executable code. Real CABT observations must replace latent predictions after every step.

Roll out in shadow mode first. Add tests for termination, depth/node/model-call/simulator-call caps, legal-list reordering, unresolved selectors, stale indices, branch observations, hidden predicate rejection, repair, timeout, and fallback. Benchmark depth 0/1/2/3, simple versus nonlinear fixtures, and 0/8/16 simulator-verification settings. Keep action selection disabled until safety and latency gates pass.
```
