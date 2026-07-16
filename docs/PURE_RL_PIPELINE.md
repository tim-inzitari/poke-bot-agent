# Pure RL Pipeline

Single active trainee on branch `cursor/pure-rl-pipeline-2d48`:

1. **Deck-agnostic core** (Stage A) with AWR, search off, full-box hardware  
2. **Warm-start `hammer-pult`** specialist → held-out gate → greedy submit  

Hope tip archived at `archive/a-new-hope-pre-pure-rl` (+ tag).

## Fatal learning contract

- `PURE_RL=1` / `TrainConfig.pure_rl_defaults()` → AWR on factorized `selected_index`  
- Soft `history_policy` CE targets **hard-fail**  
- Aux / strategy head loss weights **0**; `POKEBOT_BLACKWELL_STRATEGY_HEADS=0`  
- Collect: `mcts_sims=0`, sample actions; eval/submit: greedy  

## Full hardware (from Stage A)

| Resource | Role |
|---|---|
| CPU workers | Max sim / in-flight games (RAM + queue capped) |
| GPU1 Blackwell | Train + majority leaf replicas |
| GPU0 3080 Ti | Same-model leaf replicas |
| Overlap | Collect shard `t+1` while training shard `t` |

Profile: `poke_bot.pure_rl.hardware.full_hardware_profile()` (`PURE_RL_*` env knobs).

## Commands

```bash
# CI / wiring canary (no CABT)
python -u scripts/launch_pure_rl.py --mode core --smoke --allow-single-gpu --preflight-profile none -- \
  --iterations 2 --smoke-games 8 --heldout-games 200

# Overnight core on the training host (both GPUs visible)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
POKEBOT_BLACKWELL_STRATEGY_HEADS=0 \
python -u scripts/launch_pure_rl.py --mode core --run-name pure_rl_core_overnight -- \
  --base-checkpoint outputs/checkpoints/<seed_or_bootstrap>.pt \
  --iterations 1000 --games-per-iter 256 --heldout-games 200 --gate-wr 0.70

# After CORE_GATE_PASSED: warm-start specialist
python -u scripts/warm_start_pure_rl_specialist.py \
  --core-checkpoint outputs/pure_rl/pure_rl_core_overnight/checkpoints/iter_XXXXX.pt \
  --run-name pure_rl_hammer --archetype hammer-pult

# Specialist overnight (same full hardware)
python -u scripts/launch_pure_rl.py --mode specialist --run-name pure_rl_hammer -- \
  --base-checkpoint outputs/pure_rl/pure_rl_hammer/checkpoints/hammer-pult_warmstart.pt \
  --iterations 1000 --games-per-iter 256
```

## Gates

- Held-out official four baselines, ≥200 seat-balanced games, WR ≥ 0.70  
- Exclude `baseline_failed` forfeits (`poke_bot.pure_rl.eval_public`)  
- Abort promote on self-distill / ~zero advantage (`poke_bot.pure_rl.aborts`)  

## Layout

```
outputs/pure_rl/<run-name>/
  shards/
  checkpoints/
  metrics/
  manifest.json
  CORE_GATE_PASSED | SPECIALIST_GATE_PASSED
```
