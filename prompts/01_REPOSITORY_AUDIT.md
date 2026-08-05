# Prompt 1 — Repository Audit

```text
Read AGENTS.md and docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md.

Perform a read-only architecture audit of this repository. Do not modify production code and do not design from guessed paths.

Find and document:
- agent and Kaggle entry points;
- CABT legal-options, option-index, clone, step, observation, seed, and turn-boundary APIs;
- current observation visibility boundary and every privileged field;
- state/action tokenization and exact tensor shapes;
- current transformer/evaluator/search modules and parameter counts;
- current MCTS call graph and where repeated simulator/model calls occur;
- training, replay, evaluation, checkpoint, specialist, and config flows;
- existing tests and commands;
- current p50/p95/max decision and whole-turn latency, model calls, simulator calls, state encodes, and peak memory on representative fixtures.

Run only safe inspection commands, existing tests, and short benchmarks. Confirm or revise the claim that roughly 75 simulator calls consume the whole turn budget.

Fill every applicable field in docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md using exact paths, symbols, shapes, commands, and measured evidence. Mark truly unavailable items as BLOCKED with the reason. End with the smallest safe Phase 1 integration plan and exact proposed file list. Do not implement Phase 1 yet.
```
