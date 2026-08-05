# Prompt 6 — Validation and Ablations

```text
Read AGENTS.md and docs/poke_rlm/07_EXPERIMENTS_AND_GATES.md.

Build or run the complete evaluation matrix:
A current policy-only;
B parallel action decoder + policy;
C B + Q/value/successor/uncertainty;
D typed root plan without recursion;
E depth 1;
F depth 2;
G depth 3 audit;
H depth 2 + up to 8 verifier calls;
I depth 2 + up to 16 verifier calls;
J legacy 75-call MCTS.

Use equal whole-turn deadlines. Report parameter counts, p50/p95/max decision and turn latency, model/simulator calls, peak memory, illegal/invalid/unresolved/fallback/repair/timeout/NaN rates, paired win rate or Elo, calibration, teacher regret, and results by deck/turn complexity.

Verify that the complexity router protects simple decks and that depth 2 provides disproportionate value on nonlinear turns. Do not promote depth 3 or strong_512 without evidence. Produce a decision table for pilot_256 versus base_384 and a recommendation on whether strong_512 is justified.
```
