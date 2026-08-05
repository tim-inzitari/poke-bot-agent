# Slowking distill research lane

Research-only. No runtime / selector / serving / freeze / registration authority.

## Recommended architecture (coded)

1. Archetype-wide option-conditioned BC (`bc_stage.py`)
2. Sparse heuristic features, never serving logits (`heuristic_features.py` + surrogate)
3. IQL expectile critic + gated AWR (`iql.py`)
4. Critical-node search receipts (`critical_search.py`; mock locally, BeliefMCTS on host)
5. Search→actor distillation (`distill_search.py`)
6. Paired win gate that cannot promote on agreement alone (`eval_gate.py`)

## CLIs

```bash
# Aggregate daily distill receipts
python3 scripts/slowking_distill_aggregate_multi_day.py \
  --daily state/slowking_top_replay_distillation_2026-08-04.json \
  --window-start 2026-08-04 --window-end 2026-08-04 \
  --out state/slowking_multi_day_from_dailies.json

# Build decision JSONL from archives (host)
python3 scripts/slowking_distill_build_corpus.py \
  --archive-date /path/day.zip 2026-08-04 \
  --out-jsonl outputs/slowking_distill/decisions.jsonl

# Run stages A→E (never self-promotes)
python3 scripts/slowking_distill_run_pipeline.py \
  --decisions-jsonl outputs/slowking_distill/decisions.jsonl \
  --out-dir outputs/slowking_distill/run \
  --val-date 2026-08-04
```

Package: `poke_bot/slowking_distill/`.
