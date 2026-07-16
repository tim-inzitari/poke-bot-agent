# Pure RL Pipeline

Independent pure-RL line on branch `cursor/pure-rl-full-rebuild-2d48`
(PR vs `main` — not a Hope merge). Hope tip archived at
`archive/a-new-hope-pre-pure-rl` (+ tag).

1. **Deck-agnostic core** (Stage A) with AWR, search off, full-box hardware  
2. **Warm-start `hammer-pult`** specialist → held-out gate → greedy submit  

## Design constraint (Abhyuday / top pure-RL competitor)

> "Not alphazero style, but yes RL. The starter is terrible."

- **Not AlphaZero-style:** Overnight collect stays **search OFF** (`mcts_sims=0`).
  No MCTS visit-count policy targets, no AZ dual-net distillation into π.
  Learning is **outcome-weighted AWR** on played factorized actions + terminal
  W/L/D value — pure RL self-play volume, not search teacher cloning.
- **Starter is terrible:** Do **not** bootstrap overnight from the competition
  RL starter / sample notebook as a skill prior. Do **not** CE-clone the
  official starter policy. Prefer a **fresh small (~1–3M) random seed**
  (`pure_rl_model_config` + `build_pure_rl_model`). Any optional checkpoint
  must be a matching small arch and must **not** be the official starter
  policy as the main prior (`bootstrap_mix=0` always).

## Small model (mandatory)

Pure-RL uses `poke_bot.pure_rl.model_profile.pure_rl_model_config()` — lean
history net (~**2.4M** params default: `d_model=24`, 1/1/1 layers, `ff=48`,
`ctx=32`). Launch **fail-closes** if `sum(p.numel()) > 3.5M`. Do **not** warm-start
from Hope's d=256 primary or the competition starter.

## Fatal learning contract

- `TrainConfig.pure_rl_defaults()` → AWR on factorized `selected_index`  
- Stale (detached) value baseline; optional advantage whitening; weight clip  
- Soft `history_policy` CE targets **hard-fail**; `bootstrap_mix=0`  
- Aux / strategy head loss weights **0**; `POKEBOT_BLACKWELL_STRATEGY_HEADS=0`  
- Collect: `mcts_sims=0`, temperature sample (annealed); eval/submit: greedy  
- Fresh-data window (`PURE_RL_REPLAY_WINDOW_SHARDS`); self-distill abort  
- Opponent pool: public/roster collect + recent-self hints; official-4 held-out gate  

## Full hardware (from Stage A)

| Resource | Role |
|---|---|
| CPU workers | Max sim / in-flight games (RAM + queue capped) |
| GPU1 Blackwell | Train + majority leaf replicas |
| GPU0 3080 Ti | Same-model leaf replicas |
| Elmo + bert | Additive **whole-game** collect farms → same shards |
| Overlap | Collect shard `t+1` while training shard `t` |

One active AWR trainee on the host. Remotes maximize games / wall-time; they
are **not** a second competing trainer.

Profile: `poke_bot.pure_rl.hardware.full_hardware_profile()` (`PURE_RL_*` env knobs).  
Remote protocol: `poke_bot.remote_jobs.RemoteWorkerFarm` + `iter_additive_results`.

Default endpoints: `192.168.1.143:8765,bert.local:8766`.

## Commands

```bash
# CI / wiring canary (no CABT, no remotes)
POKEBOT_PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python \
  bash scripts/canary_pure_rl.sh

# Overnight core on the training host (both GPUs + remotes)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python

$POKEBOT_PYTHON -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name pure_rl_core_overnight_<UTC> \
  --preflight-profile none \
  --log outputs/logs/pure_rl_core.log \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  -- \
  --base-checkpoint outputs/pure_rl/<run>/checkpoints/seed.pt \
  --iterations 1000 \
  --games-per-iter 256 \
  --heldout-games 200 \
  --gate-wr 0.70

# After CORE_GATE_PASSED: warm-start specialist
python -u scripts/warm_start_pure_rl_specialist.py \
  --core-checkpoint outputs/pure_rl/pure_rl_core_overnight/checkpoints/iter_XXXXX.pt \
  --run-name pure_rl_hammer --archetype hammer-pult

# Specialist overnight (same full hardware + remotes)
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
