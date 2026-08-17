# Claim ledger and pending evidence

The rendered paper uses evidence labels instead of blank result cells. This
keeps the template honest and readable before every final evaluation exists.

## Measured and receipt-indexed

- Corpus: 20 UTC days, 24,335 games, 2,351,208 decisions, 18,412,973 option
  records, 68 acting-deck variants; 14/3/3 whole-day split with zero reported
  cross-split source/day/game/group overlap.
- Collision census: 127,641 legacy groups affecting 255,398 option records;
  repaired representation has zero unresolved actionable groups. All 99
  remaining duplicates are harmless permutations.
- Full bootstrap: 25 epochs, 57.88 million decision visits, 5,757.14 seconds,
  final semantic loss 1.7141 and semantic accuracy 42.27%. These are training
  diagnostics.
- H3 sidecar: held-out prediction and ESS/clipping non-regression only. It is
  not runtime policy evidence.
- Iteration 2: 900-game formal audit, 76.83% skill-weighted win rate, 74.39%
  recorded lower bound, but broader pipeline gate failed. Non-promotion.
- Kaggle submission 55487412: score 600.0, NO-RTP/search/MCTS-off package. One
  contextual public observation.

## Pending or explicitly unavailable

- No causal adapter-on versus adapter-off gameplay estimate is complete.
- Iteration 3 formal holdout: waived, zero games, no pass claim.
- Iteration 4: collection sealed at the snapshot, but no commit/evaluation
  result exists.
- Required legacy/H1/H3/H6 paired arms: future evidence, 8,192 matched games
  per eligible arm when capacity and receipts permit.
- Checklist gates: the branch adjudication accepted only the public-rule
  semantic projection. Checklist-provenance gates remain exact-zero/inert.
- Recursive turn planning and RTP: shadow/roadmap only, not active packaged
  action selection.

## How to update

For each new result, verify the receipt and digest, add it to
`SOURCE_MANIFEST.md`, change the evidence label and prose in the relevant
section/table, and preserve the previous disposition in version control. Never
replace "not measured" with "pass" from a waiver, configuration, or public
score alone.
