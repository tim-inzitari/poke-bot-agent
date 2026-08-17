# PokeRLM Project Instructions

## Read order

Before changing code, read:

1. `RL_TRAINING_PROTOCOL.md`
2. `config/rl_protocol.yaml`
3. `state/specialists.yaml`
4. `docs/poke_rlm/00_EXECUTIVE_DESIGN.md`
5. `docs/poke_rlm/01_ARCHITECTURE.md`
6. `docs/poke_rlm/02_TYPED_PLAN_IR.md`
7. `docs/poke_rlm/06_IMPLEMENTATION_ROADMAP.md`
8. `docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md`

If any path is absent, record that fact in the repository mapping. Do not invent its contents.

## Non-negotiable invariants

- CABT is the exact simulator and legality authority.
- Submit only a legal option index from the **current** CABT legal-action list.
- Model inputs must contain only information visible to the acting player. Opponent hand, prizes, future reveals, and privileged simulator state may never enter deployment inputs.
- The measured legacy ceiling is about 75 simulator calls for the **entire turn**, not per atomic decision. PokeRLM should normally use 0–16 online simulator calls and must never assume 75 calls are free.
- Do not restore full online MCTS as the primary planner.
- Encode the state once per turn whenever safe. Never run the full backbone once per legal action.
- Generate one conditional plan per turn and repair only when observations invalidate or materially degrade it.
- Recursion must be bounded by depth, plan-node count, subgoal count, model-call count, and wall-clock deadline.
- Recursive planner weights are shared across depth. Increasing recursion depth must not instantiate a new planner network.
- The neural model may propose actions, subgoals, and branches; exact legality, resource accounting, option-index resolution, and state transitions remain outside the neural model.
- No arbitrary Python, shell, REPL, or natural-language program execution may be emitted by the planner.
- Keep the current policy agent available behind a deterministic fallback and a feature flag until the redesign passes gates.
- Preserve exact replay reproducibility, simulator/rules/card-database versions, seeds, and observation schema versions.

## Existing training protocol remains authoritative

Do not silently change:

- exactly 25 supervised bootstrap epochs per new specialist;
- exactly 8,192 games per baseline-phase RL iteration;
- 7,168 public-mix games plus 1,024 self-play games;
- five supervised rehearsal epochs after every five completed RL epochs;
- exactly 3,000 isolated evaluation games per required evaluation cycle;
- frozen-specialist update isolation;
- research holdout exclusion rules.

Architecture experiments may add labels, losses, metrics, and shadow evaluations, but they must not mutate these numbers unless the authoritative protocol and config are explicitly updated together.

## Required workflow for every implementation phase

1. Inspect the real code and update `docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md` with exact paths and symbols.
2. Capture the current baseline: parameter count, model calls, simulator calls, p50/p95/max latency, peak memory, illegal-action rate, and win-rate/Elo evaluation command.
3. Write or update interfaces and tests before replacing behavior.
4. Add the new component behind a config flag. Default to current behavior until parity tests pass.
5. Run focused unit tests, deterministic replay tests, hidden-information leakage tests, and latency smoke tests.
6. Run shadow mode before enabling action selection.
7. Update `state/poke_rlm_redesign.yaml` or the repository-equivalent progress file with artifacts and evidence.
8. Preserve validated checkpoints and avoid unrelated refactors.

## Code expectations

- Prefer typed Python and explicit dataclasses/protocols for planner contracts.
- Keep tensor shapes in docstrings and assertions at module boundaries.
- Use batched tensor operations; no Python loop over legal actions during deployed inference.
- Use fused output projections where practical.
- Separate acting-player observations from privileged training targets in both types and storage.
- Reject invalid plans early and return a structured reason code.
- Every fallback must be deterministic, logged, and covered by a test.
- Use the repository's existing formatter, linter, type checker, test runner, and configuration conventions.
- Do not add a new framework when an existing repository abstraction is sufficient.

## Recommended initial profile

Use `config/poke_rlm_planner.example.yaml` as a design reference, not as proof of the repository's actual schema.

The recommended target is:

- profile: `base_384`
- shared parameters: approximately 35.7M total
- new planning attachment: approximately 14.5M when attached to a compatible existing encoder
- specialist adapter: approximately 0.4M–0.9M per archetype
- recursion depth: 2
- maximum neural planner calls per turn: 4
- maximum plan nodes: 32
- target online simulator calls: 0–16 per turn

Measure before scaling. Move to `strong_512` only when the base model shows persistent capacity-limited underfitting rather than data, label, optimization, or interface problems.
