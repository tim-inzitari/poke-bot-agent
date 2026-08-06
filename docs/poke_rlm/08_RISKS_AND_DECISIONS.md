# Risks, Failure Modes, and Design Decisions

## 1. Learned-model exploitation

**Failure:** The planner discovers plan fragments that look strong only because latent dynamics are wrong.

**Controls:**

- short horizons;
- structured delta supervision;
- symbolic resource ledger;
- uncertainty penalties;
- multiple teacher policies;
- sparse exact verification;
- on-policy failed-plan relabeling;
- predicted-versus-observed trace audits.

## 2. Compounding recursive error

**Failure:** Each recursive refinement builds on an incorrect assumption and produces a coherent but bad plan.

**Controls:**

- shared depth-limited planner;
- maximum depth 2 initially;
- real observations replace predictions after every CABT step;
- branch predicates and fallbacks;
- value checks against direct policy;
- stop and repair heads;
- depth-3 audit specifically for overthinking.

## 3. Invalid or stale option indices

**Failure:** A plan stores an index that no longer maps to the intended action after the legal list changes.

**Decision:** Future plan nodes store typed selectors/fingerprints. The executor resolves a **current** CABT option index immediately before submission.

## 4. Hidden-information leakage

**Failure:** Offline teacher or simulator state leaks opponent hand, prizes, or future reveals into deployment features.

**Controls:**

- distinct deployment and privileged target types;
- visibility masks;
- timestamped reconstruction;
- import/layer boundaries;
- plan predicate whitelist;
- protected split and leakage tests.

## 5. Recursion harms simple decks

**Failure:** Extra compute and plan noise reduce performance on nearly forced turns.

**Decision:** Complexity routing is a required module. Direct selection remains a first-class route and fallback.

## 6. Tail latency

**Failure:** Average latency looks acceptable while long combo turns time out.

**Controls:**

- shared deadline object;
- p95/max reporting;
- hard model/simulator/node/depth caps;
- batched candidates and same-depth subgoals;
- cache invalidation contract;
- early stop to best valid plan;
- deterministic fallback.

## 7. Training objective interference

**Failure:** Dynamics, plan, or belief losses damage a strong policy representation.

**Controls:**

- staged curriculum;
- frozen core first;
- normalized losses;
- gradient monitoring;
- lower core learning rate;
- head and loss ablations;
- broad rehearsal data.

## 8. Biased teacher plans

**Failure:** The student copies the limitations of one rollout policy or search heuristic.

**Controls:**

- teacher diversity;
- shorter rollout horizons;
- process and outcome labels kept separately;
- periodic relabeling;
- human/heuristic audits where available;
- on-policy correction queue.

## 9. Specialist fragmentation

**Failure:** Each deck becomes a separate full model and shared transfer collapses.

**Decision:** Share encoder, embeddings, action decoder, plan grammar, most dynamics, and base heads. Use small upper-block adapters and plan-priority calibration. Distill repeatable gains back to the core.

## 10. Protocol contamination

**Failure:** Architecture work silently changes the established 25-epoch bootstrap, 8,192-game iteration, rehearsal cadence, holdouts, or specialist freezing.

**Control:** Existing protocol/config/state files remain authoritative. PokeRLM state tracks architecture progress without replacing specialist state.

## 11. Parameter scaling too early

**Failure:** A 70M model masks data and target bugs while increasing latency.

**Decision:**

1. `pilot_256` for wiring;
2. `base_384` for serious strength;
3. `strong_512` only after scaling evidence.

## 12. Online simulator creep

**Failure:** “Optional verification” gradually becomes another broad search consuming the whole turn.

**Decision:** Default target 0–16 calls, shared turn budget, explicit call telemetry, and ablation against 0/8/16. The 75-call legacy figure remains a hard emergency ceiling, never the normal target.

## Decision log

| Decision | Status | Rationale |
|---|---|---|
| Use PokeRLM rather than full online MCTS | Accepted | Whole-turn simulator latency makes broad search ineffective |
| Typed plan IR, not natural language | Accepted | Exactness, safety, supervision, predictable execution |
| Shared recursive weights | Accepted | Parameter efficiency and bounded scaling |
| Depth 2 initial maximum | Accepted | Enough decomposition without uncontrolled latency/error |
| CABT exact validator/executor | Accepted | Neural model must not own game truth |
| `base_384` recommended | Accepted | Approx. 35.7M gives a strong capacity/latency compromise |
| 0–16 normal simulator calls | Accepted for pilot | Leaves time for neural planning and execution |
| Small specialist adapters | Accepted | Preserve transfer and Kaggle memory |
| Depth 3 | Experiment only | Explicit overthinking and tail-latency audit |
