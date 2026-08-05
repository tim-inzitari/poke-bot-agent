# Cursor Master Orchestrator Prompt

Paste this into Cursor Agent after copying the kit into the repository.

```text
You are implementing the PokeRLM redesign in this repository.

Read AGENTS.md, every file under docs/poke_rlm, the applicable .cursor/rules, RL_TRAINING_PROTOCOL.md, config/rl_protocol.yaml, and state/specialists.yaml before editing.

Work phase-by-phase according to docs/poke_rlm/06_IMPLEMENTATION_ROADMAP.md.

Mandatory behavior:
1. Begin with the repository audit in prompts/01_REPOSITORY_AUDIT.md.
2. Do not invent file paths, APIs, tensor shapes, or current parameter counts.
3. Fill docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md with exact evidence.
4. Preserve current production behavior behind feature flags until parity, safety, and latency gates pass.
5. CABT remains the exact legality and transition authority.
6. Never expose hidden opponent information to deployment inputs.
7. Never run the full backbone once per legal action.
8. Treat roughly 75 simulator calls as a whole-turn hard ceiling; target 0–16 for PokeRLM.
9. Recursion must be typed, bounded, weight-shared, and free of arbitrary generated code.
10. Do not change authoritative RL numeric requirements silently.
11. Run tests and benchmarks after each coherent phase.
12. Update the PokeRLM progress state with files, commands, results, parameter counts, latency, and remaining blockers.
13. Do not start expensive multi-hour or multi-million-game training from this prompt. Implement, validate, and produce the exact command/config for the next controlled run.

Continue autonomously through only those phases whose previous pass conditions are demonstrably satisfied. When a gate is not met, diagnose and fix within scope; otherwise stop with concrete evidence rather than pretending the phase passed.
```
