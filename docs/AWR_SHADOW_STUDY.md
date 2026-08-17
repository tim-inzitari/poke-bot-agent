# AWR shadow study

This study is intentionally separate from the production RL ledger. It cannot
publish weights, reload workers, edit services, or promote a candidate.

## Phase 1: beta

Prepare an immutable manifest from one parent checkpoint, the exact replay
shards, and the installed official-baseline contract:

```bash
python scripts/run_awr_shadow_study.py prepare-beta \
  --output-dir /data/pokebot-shadow/awr-beta-001 \
  --parent-checkpoint /path/to/immutable-parent.pt \
  --replay-shard /path/to/shards/iter_00005.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00005.json \
  --replay-shard /path/to/shards/iter_00006.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00006.json \
  --baseline-contract /path/to/baselines/manifest.json \
  --shadow-device cuda:0 \
  --production-device cuda:1
```

The manifest hard-locks AdamW, LR `3e-4`, weight decay `1e-4`, gradient clip
`1.0`, decision cap `8192`, temporal context `320`, one parent digest, one
replay-set digest, and one split/order seed. Its only Phase-1 axis is AWR beta:
`0.5`, `0.75`, and `1.0`.

Every replay shard must have its matching `poke_bot.completed_collection/v1`
receipt. Preparation reconciles shard digest, byte size, game/decision counts,
zero-drop feature cache, and context 320 before accepting the boundary.

Candidate fitting is an explicit second command and must run on the manifest's
separate shadow device. The launcher checks `nvidia-smi` and fails closed if a
foreign compute process is already using that GPU:

```bash
python scripts/run_awr_shadow_study.py run-beta \
  --manifest /data/pokebot-shadow/awr-beta-001/study_manifest.json \
  --device cuda:0 \
  --ack-shadow-only
```

Each candidate receipt records clip fraction, ESS and effective fraction,
p50/p95/maximum AWR weights, policy/value/all auxiliary losses, update norm,
unique games, unique episodes, unique decisions, and the exact split/batch-order
digests. Source replay status writes are disabled while the shadow loader reads
production shards.

Evaluation rows must cover the manifest's identical requested seeds,
opponents, and seats for every beta candidate. Finalization rejects missing,
duplicate, invalid, wrong-checkpoint, or schedule-mismatched rows and uses
Bonferroni-spent sequential Hoeffding intervals:

```bash
python scripts/run_awr_shadow_study.py finalize-beta \
  --manifest /data/pokebot-shadow/awr-beta-001/study_manifest.json \
  --evaluation-rows /data/pokebot-shadow/awr-beta-001/evaluation_rows.json
```

The official engine lacks a native RNG seed API. The requested Python/Torch job
seeds are matched, but reports explicitly set `pairing_claimed=false`; the
confidence calculation does not rely on paired engine randomness.

## Phase 2: guarded matrix

Phase 2 cannot be prepared from the training-only report. It requires the
immutable, complete matched-evaluation report and exactly four replay shards:

```bash
python scripts/run_awr_shadow_study.py prepare-stage2 \
  --beta-manifest /data/pokebot-shadow/awr-beta-001/study_manifest.json \
  --final-beta-report /data/pokebot-shadow/awr-beta-001/beta_report.final.json \
  --output-dir /data/pokebot-shadow/awr-stage2-001 \
  --replay-shard /path/to/shards/iter_00003.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00003.json \
  --replay-shard /path/to/shards/iter_00004.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00004.json \
  --replay-shard /path/to/shards/iter_00005.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00005.json \
  --replay-shard /path/to/shards/iter_00006.jsonl \
  --replay-receipt /path/to/collection_receipts/iter_00006.json
```

That manifest stages, but does not start, the `2e-4` LR and max-AWR-weight `10`
axes with `4 shards × 1 epoch` versus `2 shards × 2 epochs`. It reports unique
game/decision counts and total decision exposures for both replay profiles.

Game-balanced loss, a BCE win-probability head, and PSRO mixtures remain
disabled proposal-only flags. They are never included in either study and have
no automatic activation path.
