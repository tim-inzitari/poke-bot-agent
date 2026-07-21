# Active Alakazam Gate

This file is the durable operator memory for the active training objective.

- Active gate ID: `alakazam-strong-public-roster-v1`
- Gate roster: the eight package-deduplicated strong public agents in
  `ops/alakazam_gate_program_v1.json`
- Exact allocation: 250 games per public opponent, 125 games in each seat,
  for 2,000 gate games total
- Gate policy: greedy, one pinned checkpoint digest, fixed disjoint seeds,
  every matchup complete, and no early-stop or partial gate result
- The original four baselines (Iono, Dragapult ex, Mega Abomasnow ex, and
  Mega Lucario ex) are 250-game-per-opponent research controls only. Their
  gate weight is exactly zero.
- The accepted 55.50% original-baseline result remains an immutable protected
  checkpoint and non-regression anchor. It is not the current gate roster.
- Sampled public-mix games are training diagnostics only. They never count as
  exact strong-public holdout evidence.

Any trainer, dashboard, evaluator, restart script, or future agent that calls
the original four-baseline evaluation "the gate" is violating this contract.
The dashboard must show strong-public exact results in the active-gate panel
and original-four results in a separately labeled research-control panel.

Submission policy invariant: whenever `SelectContext.IS_FIRST` asks whether to
go first, select the unique `OptionType.YES`. This is a packaging/runtime rule;
it does not authorize a Kaggle submission.
