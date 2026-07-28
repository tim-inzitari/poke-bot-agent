# Controller synchronization handoff

Observed at: `2026-07-28T15:46:30Z`

This is a coordination snapshot, not a new source of truth. Every controller
must read `GOAL.md` completely and then use its canonical sources, immutable
receipts, runtime selector, and managed-service state. If this snapshot and a
live receipt disagree, the live receipt wins.

## Current ownership boundary

- Active specialist: `thwackey`
- Canonical selector:
  `/home/inzi/.config/pokebot/specialist_runtime.env`
- Runtime root:
  `/home/inzi/poke-bot-agent-deployments/dudunsparce-decision-fusion-v1`
- Managed trainer:
  `pokebot-pure-rl-trevenant-staged.service`
- Active run:
  `pure_rl_thwackey_temporal1_8k_v1_20260723`
- Observed trainer state: `active/running`, PID `2690510`, zero restarts
- Observed progress: iteration 1 is receipt-committed; its exact 3,000-game
  premium holdout recorded 53.73% skill-weighted win rate and passed the
  audit, but did not pass the full gate. Iteration 2 started normally and was
  collecting self-play at 416/1,024 games at this snapshot.

Do not stop, restart, replace, or mutate the healthy Thwackey trainer. Do not
change the selector or create a duplicate trainer.

## Next-specialist preparation

- Selected successor: `hammer-pult`
- Prestage receipt:
  `/home/inzi/poke-bot-agent/outputs/state/next-specialist-prestage-v1.json`
- Only remaining blocker:
  `current_deck_guide_corpus_binding_not_ready`
- Hammer-Pult terminal gate-handler preflight: ready
- Exact representative: ready
- Ordinary latest-20 expert corpus: ready
- Guide contract, teacher, and expert write-up: ready
- Current guide window is running on Elmo with four independent days in
  parallel. Four of twenty days are complete and checksum-bound; days
  2026-07-08 through 2026-07-11 are active. The final ready receipt does not
  exist yet, so the Inzi promotion timer is correctly blocked.

The armed chain is:

1. Elmo guide window
2. Elmo corpus finalizer
3. Inzi promotion timer
4. Prestager and CPU pack
5. Exact terminal gate-handler preflight
6. Freeze and immutable registration
7. One-shot Kaggle queue authorization
8. Cumulative-core attempt
9. Atomic successor selector commit and managed start

Do not manually perform any later node while an earlier receipt is incomplete.

## Kaggle state

- Most recent accepted specialist submission:
  `rockets-mewtwo`, submission ID `55056041`
- Submitted at: `2026-07-28 13:24:45.570000 UTC`
- Kaggle submissions are asynchronous and non-blocking.
- Never create a duplicate authorization for an already queued or accepted
  frozen checkpoint.

## Transition reliability changes already pushed

- `2baee08` — one specialist runtime registry
- `16a5ada` — terminal handoff prevalidation during prestage
- `ce278da` — successor selection bound to canonical mutable state
- `8185cff` — four-day Hammer-Pult guide prestage

Branch: `codex/neural-baseline-gate-v13-20260721`

The takeover validation ran the goal-gateway, runtime registration,
sequential-handoff, cycle-handoff, transition-graph, guide-snapshot, corpus
import, and next-specialist prestage tests in the canonical Inzi environment:
68 tests passed.

## Refresh commands

```bash
ssh inzi@192.168.1.151 \
  'systemctl --user show pokebot-pure-rl-trevenant-staged.service \
  -p ActiveState -p SubState -p MainPID -p NRestarts'

ssh inzi@192.168.1.151 \
  'cat /home/inzi/poke-bot-agent/outputs/state/next-specialist-prestage-v1.json'

ssh elmo \
  'cat /mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/daily/current-deck-guides-v1/hammer-pult/status/window.json'
```
