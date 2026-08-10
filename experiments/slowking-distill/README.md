# Slowking distill research lane

Research-only. No runtime / selector / serving / freeze / registration authority.

## Recommended architecture (coded)

1. Archetype-wide option-conditioned BC (`bc_stage.py`)
2. Sparse heuristic features, never serving logits (`heuristic_features.py` + surrogate)
3. IQL expectile critic + gated AWR (`iql.py`)
4. Critical-node search receipts (`critical_search.py` + `belief_search_backend.py`; mock locally, BeliefMCTS on host)
5. Search→actor distillation (`distill_search.py`)
6. Population self-play vs frozen opponents (`self_play.py`)
7. Gated runtime + PolicyAgent bridge (`runtime.py`, `policy_bridge.py`; default off)
8. Paired win gate + fail-closed promotion (`eval_gate.py`, `promotion.py`)

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

# Run stages A→E (+ D self-play); never self-promotes
python3 scripts/slowking_distill_run_pipeline.py \
  --decisions-jsonl outputs/slowking_distill/decisions.jsonl \
  --out-dir outputs/slowking_distill/run \
  --val-date 2026-08-04

# Stage D only
python3 scripts/slowking_distill_self_play.py \
  --actor-checkpoint outputs/slowking_distill/run/stage_e/stage_e_search_distilled_actor.pt \
  --out-dir outputs/slowking_distill/self_play
```

## Runtime (default off)

```bash
export POKEBOT_SLOWKING_DISTILL_ENABLED=1
export POKEBOT_SLOWKING_DISTILL_MODE=shadow   # or active
export POKEBOT_SLOWKING_DISTILL_ACTOR_CKPT=/path/to/stage_e_actor.pt
```

Dispatch when `MODE=active`: MCTS → PokeRLM active → Slowking distill → RTP → greedy.
Shadow records telemetry only. Heuristic scores never enter serving logits.
`promotion.py` always leaves `promoted=false`.

Package: `poke_bot/slowking_distill/`.
