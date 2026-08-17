# Claims and update rules

## Safe mathematical claims at the 2026-08-14 snapshot

- The simulator owns the legal action set. The policy ranks only the current
  factorized legal candidates.
- Runtime inputs are restricted to acting-player-visible information. Opponent
  hand identities, deck order, unrevealed Prize identities, and future
  transitions are excluded.
- For n menu options and legal length l through u, the complete ordered action
  count is the sum of n!/(n-k)! over k from l through u.
- The semantic projection is additive to the preserved option representation.
  Its implemented residual cap is 0.25, and a zero gate returns the baseline
  tensor by identity.
- The collision census contains 18,412,973 option records. The legacy key
  affected 255,398 records in 127,641 groups; the repaired key left zero
  unresolved actionable groups. Its 99 duplicate groups were harmless
  permutations.
- H3 is trainer-only. Its exact actor coefficient is 0.025, its mask requires
  three complete causal same-seat segments, and H1/H6/H12 actor coefficients
  are zero in H3 mode.
- Offline H3 action-value MSE was 0.0040745 versus 0.0046184 for the
  empirical-mean baseline on 358,344 available targets. This is prediction
  evidence, not gameplay strength.
- On the same 2,146,670-decision replay membership, the H3 weighting check
  changed ESS fraction from 0.251337 to 0.251758 and clip fraction from
  0.076078 to 0.075819. This is weight-stability evidence only.
- Powerful Hand places two damage counters per hand card. In the absence of a
  visible prevention effect, the knockout threshold is ceiling(remaining HP
  divided by 20); 330 HP therefore requires 17 cards.

## Claims to avoid

- Do not describe the model as learning, creating, or filtering legality.
- Do not claim hidden Prize inference, opponent-hand access, or future-state
  access.
- Do not say H3 supplies a runtime action, changes the Kaggle policy state, or
  proves a win-rate gain.
- Do not claim every checklist or metadata branch affects logits. Unsupported
  branches remain exact-zero or trace-only.
- Do not use one leaderboard score as causal evidence for any equation or
  architecture component.
- Do not describe the guide-level route formulation as an online global
  optimizer or search algorithm; it is a mathematical statement of strategy
  constraints.

## Update after new evidence

When a new component evaluation seals:

1. verify its immutable receipt and digest;
2. update SOURCE_MANIFEST.md;
3. change only the relevant prose and table row;
4. preserve the distinction between prediction, weighting, representation,
   and gameplay evidence;
5. rebuild and recount the complete rendered PDF.
